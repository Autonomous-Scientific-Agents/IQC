"""Report generation for IQC."""

import dataclasses
from typing import Optional, List, Dict, Any, Tuple
import json
from rdkit import Chem
from rdkit.Chem import Draw
import io
import base64
import logging
import tomli
import os
from . import rdkittools
from .rdkittools import xyz_to_mol


@dataclasses.dataclass
class MoleculeInfo:
    """Information about a molecule."""

    # Basic identifiers
    formula: str = ""
    smiles: Optional[str] = None
    inchi: Optional[str] = None
    inchikey: Optional[str] = None
    iupac_name: Optional[str] = None
    pubchem_cid: Optional[str] = None  # Added PubChem CID field
    comment: Optional[str] = None  # Added comment field from XYZ file

    # Physical properties
    molecular_weight: Optional[float] = None

    # Structural properties
    num_atoms: Optional[int] = None
    num_bonds: Optional[int] = None
    num_heavy_atoms: Optional[int] = None
    num_hydrogens: Optional[int] = None
    max_bond_order: Optional[float] = None

    # Electronic properties
    num_electrons: Optional[int] = None
    num_radical_electrons: Optional[int] = None
    charge: Optional[int] = 0

    # Symmetry information
    symmetry_point_group: Optional[str] = None

    # Visualization
    image_base64: Optional[str] = None
    xyz_data: Optional[str] = None  # For 3D visualization

    @classmethod
    def from_rdkit_mol(
        cls,
        mol: Chem.Mol,
        charge: int = 0,
        xyz_data: Optional[str] = None,
        comment: Optional[str] = None,
        disable_pubchem: bool = False,
    ) -> "MoleculeInfo":
        """Create a MoleculeInfo object from an RDKit molecule.

        Parameters
        ----------
        mol : Chem.Mol
            The RDKit molecule.
        charge : int, optional
            The charge of the molecule, by default 0.
        xyz_data : Optional[str], optional
            The XYZ string data for 3D visualization, by default None.
        comment : Optional[str], optional
            The comment line from the XYZ file, by default None.
        disable_pubchem : bool, optional
            If True, skip PubChem lookups, by default False.

        Returns
        -------
        MoleculeInfo
            The molecule information.
        """
        if mol is None:
            # If mol is None, but we have xyz_data, we might still want to create a minimal entry
            # For now, returning an empty cls() or one with just xyz_data if needed.
            # However, the current logic relies on 'mol' for most fields.
            # This case might need further refinement based on desired behavior if mol conversion fails but XYZ is present.
            return cls(xyz_data=xyz_data, comment=comment)

        # Generate image and convert to base64
        img = Draw.MolToImage(mol)
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        # Calculate spin from number of electrons and radical electrons
        num_electrons_val = rdkittools.get_num_electrons(mol) - charge
        num_radical_electrons_val = rdkittools.get_num_radical_electrons(mol)

        # Get SMILES and InChI for PubChem lookup
        smiles = rdkittools.get_smiles(mol)
        inchi = rdkittools.get_inchi(mol)

        # Initialize PubChem related fields
        pubchem_cid = None
        iupac_name = None

        # Import the pubchemtools module and perform lookups if not disabled
        if not disable_pubchem:
            try:
                from . import pubchemtools

                # Try to get PubChem CID from InChI or SMILES
                if inchi:
                    try:
                        # Use the singular function first
                        cid = pubchemtools.get_cid_from_inchi(inchi)
                        if cid:
                            pubchem_cid = cid
                        else:
                            # If no primary CID, try getting all CIDs
                            cids = pubchemtools.get_cids_from_inchi(inchi)
                            if cids:
                                pubchem_cid = ",".join(cids)
                    except Exception as e:
                        logging.warning(f"Failed to get PubChem CID from InChI: {e}")

                if not pubchem_cid and smiles:
                    try:
                        # Use the singular function first
                        cid = pubchemtools.get_cid_from_smiles(smiles)
                        if cid:
                            pubchem_cid = cid
                        else:
                            # If no primary CID, try getting all CIDs
                            cids = pubchemtools.get_cids_from_smiles(smiles)
                            if cids:
                                pubchem_cid = ",".join(cids)
                    except Exception as e:
                        logging.warning(f"Failed to get PubChem CID from SMILES: {e}")

                # Get IUPAC name if we have a CID
                if pubchem_cid:
                    try:
                        # Try the first CID if we have multiple
                        first_cid = pubchem_cid.split(",")[0]
                        iupac_name = pubchemtools.get_iupac_name(first_cid)

                        # If there are multiple CIDs and the first one didn't give a name, try others
                        if not iupac_name and "," in pubchem_cid:
                            cid_list = pubchem_cid.split(",")
                            # Try each CID until we find a valid IUPAC name
                            for cid in cid_list[
                                1:
                            ]:  # Skip the first one we already tried
                                name = pubchemtools.get_iupac_name(cid)
                                if name:
                                    iupac_name = name
                                    break
                    except Exception as e:
                        logging.warning(f"Failed to get IUPAC name from PubChem: {e}")
            except ImportError:
                logging.warning(
                    "pubchemtools module not available, skipping PubChem lookups"
                )
            except Exception as e:
                logging.warning(f"Error in PubChem operations: {e}")
        else:
            logging.info("PubChem lookups disabled by user request")

        return cls(
            formula=rdkittools.get_formula(mol),
            smiles=smiles,
            inchi=inchi,
            inchikey=rdkittools.get_inchikey(mol),
            iupac_name=iupac_name,
            pubchem_cid=pubchem_cid,
            comment=comment,
            molecular_weight=rdkittools.get_mol_weight(mol),
            num_atoms=rdkittools.get_num_atoms(mol),
            num_bonds=rdkittools.get_num_bonds(mol),
            num_heavy_atoms=rdkittools.get_num_heavy_atoms(mol),
            num_hydrogens=rdkittools.get_num_hydrogens(mol),
            max_bond_order=rdkittools.get_max_bond_order(mol),
            num_electrons=num_electrons_val,
            num_radical_electrons=num_radical_electrons_val,
            charge=charge,
            image_base64=img_str,
            xyz_data=xyz_data,  # Store xyz_data
        )

    def generate_html_report(self) -> str:
        """Generate an HTML report for the molecule.

        Returns
        -------
        str
            The HTML report as a string.
        """
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Molecule Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .container { max-width: 800px; margin: 0 auto; }
                .header { text-align: center; margin-bottom: 20px; }
                .molecule-image { text-align: center; margin: 20px 0; }
                .info-section { margin: 20px 0; }
                table { width: 100%; border-collapse: collapse; }
                th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Molecule Report</h1>
                </div>
        """

        # Add molecule image if available
        if self.image_base64:
            html += f"""
                <div class="molecule-image">
                    <img src="data:image/png;base64,{self.image_base64}" alt="Molecule Structure">
                </div>
            """

        # Add basic information
        html += """
                <div class="info-section">
                    <h2>Basic Information</h2>
                    <table>
                        <tr><th>Property</th><th>Value</th></tr>
        """

        # Add all available properties
        if self.formula:
            html += f"<tr><td>Formula</td><td>{self.formula}</td></tr>"
        if self.smiles:
            html += f"<tr><td>SMILES</td><td>{self.smiles}</td></tr>"
        if self.inchi:
            html += f"<tr><td>InChI</td><td>{self.inchi}</td></tr>"
        if self.inchikey:
            html += f"<tr><td>InChIKey</td><td>{self.inchikey}</td></tr>"
        if self.molecular_weight:
            html += f"<tr><td>Molecular Weight</td><td>{self.molecular_weight:.4f}</td></tr>"
        if self.charge is not None:
            html += f"<tr><td>Charge</td><td>{self.charge}</td></tr>"
        if self.num_atoms:
            html += f"<tr><td>Number of Atoms</td><td>{self.num_atoms}</td></tr>"
        if self.num_bonds:
            html += f"<tr><td>Number of Bonds</td><td>{self.num_bonds}</td></tr>"
        if self.num_heavy_atoms:
            html += f"<tr><td>Number of Heavy Atoms</td><td>{self.num_heavy_atoms}</td></tr>"
        if self.num_hydrogens:
            html += (
                f"<tr><td>Number of Hydrogens</td><td>{self.num_hydrogens}</td></tr>"
            )
        if self.max_bond_order:
            html += (
                f"<tr><td>Maximum Bond Order</td><td>{self.max_bond_order}</td></tr>"
            )
        if self.num_electrons:
            html += (
                f"<tr><td>Number of Electrons</td><td>{self.num_electrons}</td></tr>"
            )
        if self.num_radical_electrons:
            html += f"<tr><td>Number of Radical Electrons</td><td>{self.num_radical_electrons}</td></tr>"

        # Close the table and HTML
        html += """
                    </table>
                </div>
            </div>
        </body>
        </html>
        """

        return html


def generate_multi_molecule_html_report(
    molecules: list[MoleculeInfo],
    include_properties: Optional[list[str]] = None,
    exclude_properties: Optional[list[str]] = None,
    enable_2d_view: bool = True,  # Parameter for 2D visualization
    enable_3d_view: bool = False,  # Parameter for 3D visualization
    collapsed_by_default: bool = True,  # Parameter to control if sections start collapsed
) -> str:
    """Generate an HTML report for a list of molecules, with property filtering and visualization options.

    Parameters
    ----------
    molecules : list[MoleculeInfo]
        A list of MoleculeInfo objects.
    include_properties : Optional[list[str]], optional
        A list of property names to include. If None, all available properties are considered.
        Available properties: 'formula', 'smiles', 'inchi', 'inchikey', 'iupac_name',
        'molecular_weight', 'num_atoms', 'num_bonds', 'num_heavy_atoms', 'num_hydrogens',
        'max_bond_order', 'num_electrons', 'num_radical_electrons', 'charge', 'symmetry_point_group'.
    exclude_properties : Optional[list[str]], optional
        A list of property names to exclude. Applied after include_properties.
    enable_2d_view : bool, optional
        Whether to include a 2D image of the molecule, by default True.
    enable_3d_view : bool, optional
        Whether to include a 3D view of the molecule, by default False.
    collapsed_by_default : bool, optional
        Whether molecule sections should be collapsed by default, by default True.
        This is helpful for reports with many molecules to improve performance.

    Returns
    -------
    str
        The HTML report as a string.
    """
    # Determine all valid property names from MoleculeInfo, excluding image_base64 and xyz_data for the table
    all_possible_props = [
        f.name
        for f in dataclasses.fields(MoleculeInfo)
        if f.name not in ["image_base64", "xyz_data"]
    ]

    # Filter properties to display
    if include_properties is not None:
        # Start with explicitly included properties that are valid
        properties_to_display_names = [
            p for p in include_properties if p in all_possible_props
        ]
    else:
        # If no include list, start with all possible properties
        properties_to_display_names = all_possible_props

    if exclude_properties is not None:
        # Remove any excluded properties
        properties_to_display_names = [
            p for p in properties_to_display_names if p not in exclude_properties
        ]

    # Map internal property names to human-readable names for the report
    property_display_map = {
        "comment": "Comment",  # Add comment as the first property
        "formula": "Formula",
        "smiles": "SMILES",
        "inchi": "InChI",
        "inchikey": "InChIKey",
        "iupac_name": "IUPAC Name",
        "pubchem_cid": "PubChem CID",
        "molecular_weight": "Molecular Weight",
        "num_atoms": "Number of Atoms",
        "num_bonds": "Number of Bonds",
        "num_heavy_atoms": "Number of Heavy Atoms",
        "num_hydrogens": "Number of Hydrogens",
        "max_bond_order": "Maximum Bond Order",
        "num_electrons": "Number of Electrons",
        "num_radical_electrons": "Number of Radical Electrons",
        "charge": "Charge",
        "symmetry_point_group": "Symmetry Point Group",
    }

    # Ensure comment appears first if it exists in properties_to_display_names
    if "comment" in properties_to_display_names:
        properties_to_display_names.remove("comment")
        properties_to_display_names.insert(0, "comment")

    # Prepare JSON data structure for molecules
    molecules_data = []

    for i, molecule_info in enumerate(molecules):
        molecule_id_name = (
            molecule_info.formula if molecule_info.formula else f"Molecule {i+1}"
        )
        molecule_id_display = (
            molecule_info.iupac_name if molecule_info.iupac_name else molecule_id_name
        )

        # Build property list
        properties = []
        for prop_name in properties_to_display_names:
            value = getattr(molecule_info, prop_name, None)
            if value is not None and value != "":  # Also check for empty strings
                display_name = property_display_map.get(
                    prop_name, prop_name.replace("_", " ").title()
                )
                if isinstance(value, float):
                    formatted_value = f"{value:.4f}"
                else:
                    formatted_value = str(value)
                properties.append({"name": display_name, "value": formatted_value})

        # Prepare molecule data
        molecule_data = {
            "id": i,
            "name": molecule_id_display,
            "formula": molecule_info.formula or "",
            "properties": properties,
        }

        # Add visualization data
        if molecule_info.image_base64:
            molecule_data["image_base64"] = molecule_info.image_base64
        if molecule_info.xyz_data:
            molecule_data["xyz_data"] = molecule_info.xyz_data

        molecules_data.append(molecule_data)

    # Create HTML with embedded JSON data and JavaScript for dynamic rendering
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Multi-Molecule Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .container {{ max-width: 95%; margin: 0 auto; }}
            .header {{ text-align: center; margin-bottom: 20px; }}
            .molecule-section {{ border: 1px solid #eee; padding: 15px; margin-bottom: 10px; border-radius: 5px; background-color: #f9f9f9; }}
            .molecule-header {{ font-size: 1.2em; margin-bottom: 10px; color: #333; cursor: pointer; display: flex; justify-content: space-between; align-items: center; }}
            .molecule-header::after {{ content: '▼'; transition: transform 0.3s; }}
            .molecule-header.collapsed::after {{ content: '▶'; }}
            .molecule-content {{ transition: max-height 0.3s ease-out, opacity 0.3s; }}
            .molecule-content.collapsed {{ max-height: 0; overflow: hidden; opacity: 0; }}
            .molecule-image {{ text-align: center; margin: 10px 0; }}
            .molecule-image img {{ max-width: 100%; height: auto; border: 1px solid #ddd; padding: 5px; background-color: white; }}
            .viewer-3d-container {{ height: 400px; width: 100%; margin: 20px 0; border: 1px solid #ccc; position: relative; }}
            .info-section {{ margin: 20px 0; }}
            table {{ width: 100%; border-collapse: collapse; }}
            th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #e9e9e9; color: #333; }}
            tr:nth-child(even) {{ background-color: #fdfdfd; }}
            .controls {{ margin-bottom: 20px; display: flex; gap: 10px; justify-content: center; }}
            .control-button {{ padding: 8px 15px; background-color: #4a4a4a; color: white; border: none; border-radius: 4px; cursor: pointer; }}
            .control-button:hover {{ background-color: #666; }}
            .molecule-count {{ text-align: center; margin-bottom: 10px; font-size: 0.9em; color: #666; }}
            .visualization-container {{ display: flex; flex-direction: column; gap: 20px; }}
        </style>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.4/3Dmol-min.js"></script>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Multi-Molecule Report</h1>
            </div>
            <div class="molecule-count">
                Showing {len(molecules)} structures
            </div>
            <div class="controls">
                <button id="expand-all" class="control-button">Expand All</button>
                <button id="collapse-all" class="control-button">Collapse All</button>
                <button id="initialize-all-3d" class="control-button">Initialize All 3D Views</button>
            </div>
            <div id="molecules-container"></div>
        </div>
        
        <script id="molecule-data" type="application/json">
        {json.dumps(molecules_data)}
        </script>
        
        <script>
        document.addEventListener('DOMContentLoaded', function() {{
            const enable2dView = {str(enable_2d_view).lower()};
            const enable3dView = {str(enable_3d_view).lower()};
            const collapsedByDefault = {str(collapsed_by_default).lower()};
            const moleculesContainer = document.getElementById('molecules-container');
            const moleculesData = JSON.parse(document.getElementById('molecule-data').textContent);
            const viewerStates = new Map(); // Track viewer initialization state
            
            if (moleculesData.length === 0) {{
                moleculesContainer.innerHTML = "<p>No molecules provided for the report.</p>";
                return;
            }}
            
            // Create all molecule sections
            moleculesData.forEach((molecule, index) => {{
                // Create molecule section
                const section = document.createElement('div');
                section.className = 'molecule-section';
                section.setAttribute('data-id', molecule.id);
                
                // Add header with click behavior for collapse/expand
                const header = document.createElement('div');
                header.className = collapsedByDefault ? 'molecule-header collapsed' : 'molecule-header';
                header.textContent = `${{molecule.name}} (Index: ${{molecule.id}})`;
                header.addEventListener('click', () => toggleSection(section));
                section.appendChild(header);
                
                // Content container (will be hidden/shown)
                const content = document.createElement('div');
                content.className = collapsedByDefault ? 'molecule-content collapsed' : 'molecule-content';
                
                // Create a visualization container to hold both 2D and 3D views if needed
                const visualizationContainer = document.createElement('div');
                visualizationContainer.className = 'visualization-container';
                
                // Add 2D view if enabled and available
                if (enable2dView && molecule.image_base64) {{
                    const imageContainer = document.createElement('div');
                    imageContainer.className = 'molecule-image';
                    
                    const label = document.createElement('p');
                    label.style.fontStyle = 'italic';
                    label.style.fontSize = '0.9em';
                    label.style.marginBottom = '5px';
                    label.textContent = '2D Structure';
                    imageContainer.appendChild(label);
                    
                    const img = document.createElement('img');
                    img.src = `data:image/png;base64,${{molecule.image_base64}}`;
                    img.alt = `2D Structure of ${{molecule.name}}`;
                    imageContainer.appendChild(img);
                    
                    visualizationContainer.appendChild(imageContainer);
                }} else if (enable2dView) {{
                    const noDataMsg = document.createElement('p');
                    noDataMsg.innerHTML = '<em>2D image not available.</em>';
                    visualizationContainer.appendChild(noDataMsg);
                }}
                
                // Add 3D view if enabled and available 
                if (enable3dView && molecule.xyz_data) {{
                    viewerStates.set(molecule.id, {{
                        initialized: false,
                        data: molecule.xyz_data
                    }});
                    
                    // Create viewer container that will be populated on demand
                    const viewerContainer = document.createElement('div');
                    viewerContainer.className = 'molecule-image';
                    
                    const label = document.createElement('p');
                    label.style.fontStyle = 'italic';
                    label.style.fontSize = '0.9em';
                    label.style.marginBottom = '5px';
                    label.textContent = '3D Structure';
                    viewerContainer.appendChild(label);
                    
                    const viewerDiv = document.createElement('div');
                    viewerDiv.id = `viewer3d-${{molecule.id}}`;
                    viewerDiv.className = 'viewer-3d-container';
                    viewerContainer.appendChild(viewerDiv);
                    
                    visualizationContainer.appendChild(viewerContainer);
                }} else if (enable3dView) {{
                    const noDataMsg = document.createElement('p');
                    noDataMsg.innerHTML = '<em>3D view enabled, but XYZ data is missing for this molecule.</em>';
                    visualizationContainer.appendChild(noDataMsg);
                }}
                
                // Add visualization container to content if it has children
                if (visualizationContainer.children.length > 0) {{
                    content.appendChild(visualizationContainer);
                }} else {{
                    const noVisualsMsg = document.createElement('p');
                    noVisualsMsg.innerHTML = '<em>No visualizations available or enabled.</em>';
                    content.appendChild(noVisualsMsg);
                }}
                
                // Add properties table
                if (molecule.properties && molecule.properties.length > 0) {{
                    const infoSection = document.createElement('div');
                    infoSection.className = 'info-section';
                    
                    const heading = document.createElement('h2');
                    heading.textContent = 'Properties';
                    infoSection.appendChild(heading);
                    
                    const table = document.createElement('table');
                    const headerRow = document.createElement('tr');
                    
                    const thProperty = document.createElement('th');
                    thProperty.textContent = 'Property';
                    headerRow.appendChild(thProperty);
                    
                    const thValue = document.createElement('th');
                    thValue.textContent = 'Value';
                    headerRow.appendChild(thValue);
                    
                    table.appendChild(headerRow);
                    
                    molecule.properties.forEach(prop => {{
                        const row = document.createElement('tr');
                        
                        const tdProperty = document.createElement('td');
                        tdProperty.textContent = prop.name;
                        row.appendChild(tdProperty);
                        
                        const tdValue = document.createElement('td');
                        tdValue.textContent = prop.value;
                        row.appendChild(tdValue);
                        
                        table.appendChild(row);
                    }});
                    
                    infoSection.appendChild(table);
                    content.appendChild(infoSection);
                }} else {{
                    const noPropsMsg = document.createElement('p');
                    noPropsMsg.textContent = 'No properties selected for display.';
                    content.appendChild(noPropsMsg);
                }}
                
                // Add content to section
                section.appendChild(content);
                
                // Add section to container
                moleculesContainer.appendChild(section);
                
                // Initialize viewer immediately if section is expanded on load
                if (!collapsedByDefault && enable3dView && viewerStates.has(molecule.id)) {{
                    initializeViewer(molecule.id);
                }}
            }});
            
            // Setup control buttons
            document.getElementById('expand-all').addEventListener('click', expandAll);
            document.getElementById('collapse-all').addEventListener('click', collapseAll);
            document.getElementById('initialize-all-3d').addEventListener('click', initializeAll3D);
            
            // Function to toggle section collapse state
            function toggleSection(section) {{
                const header = section.querySelector('.molecule-header');
                const content = section.querySelector('.molecule-content');
                const isCollapsed = header.classList.contains('collapsed');
                
                // Toggle classes
                header.classList.toggle('collapsed');
                content.classList.toggle('collapsed');
                
                // If expanding and has uninitialized 3D viewer, initialize it
                if (isCollapsed && enable3dView) {{
                    const moleculeId = parseInt(section.getAttribute('data-id'));
                    if (viewerStates.has(moleculeId) && !viewerStates.get(moleculeId).initialized) {{
                        initializeViewer(moleculeId);
                    }}
                }}
            }}
            
            // Function to initialize a specific 3D viewer
            function initializeViewer(moleculeId) {{
                if (!viewerStates.has(moleculeId) || viewerStates.get(moleculeId).initialized) return;
                
                const viewerState = viewerStates.get(moleculeId);
                const viewerElement = document.getElementById(`viewer3d-${{moleculeId}}`);
                
                if (!viewerElement) return;
                
                try {{
                    const viewer = $3Dmol.createViewer(viewerElement, {{backgroundColor: 'white'}});
                    viewer.addModel(viewerState.data, 'xyz');
                    viewer.setStyle({{}}, {{stick: {{}}}});
                    viewer.zoomTo();
                    viewer.render();
                    
                    // Mark as initialized
                    viewerState.initialized = true;
                    viewerStates.set(moleculeId, viewerState);
                }} catch (e) {{
                    console.error(`Error initializing viewer for molecule ${{moleculeId}}:`, e);
                    viewerElement.innerHTML = 
                        `<p style="color: red; padding: 10px;">Error initializing 3D viewer: ${{e.message}}</p>`;
                }}
            }}
            
            // Function to expand all sections
            function expandAll() {{
                document.querySelectorAll('.molecule-header').forEach(header => {{
                    header.classList.remove('collapsed');
                }});
                document.querySelectorAll('.molecule-content').forEach(content => {{
                    content.classList.remove('collapsed');
                }});
                
                // Don't automatically initialize all viewers to avoid memory issues
                // User can click the dedicated button if they want to initialize all
            }}
            
            // Function to collapse all sections
            function collapseAll() {{
                document.querySelectorAll('.molecule-header').forEach(header => {{
                    header.classList.add('collapsed');
                }});
                document.querySelectorAll('.molecule-content').forEach(content => {{
                    content.classList.add('collapsed');
                }});
            }}
            
            // Function to initialize all 3D viewers
            function initializeAll3D() {{
                if (!enable3dView) return;
                
                const confirmation = confirm(
                    "This will initialize " + viewerStates.size + " 3D viewers, which may use significant memory. Continue?"
                );
                
                if (confirmation) {{
                    viewerStates.forEach((state, id) => {{
                        if (!state.initialized) {{
                            initializeViewer(id);
                        }}
                    }});
                    
                    // Expand all sections to show the viewers
                    expandAll();
                }}
            }}
        }});
        </script>
    </body>
    </html>
    """

    return html


def write_report(report: str, output_file: str):
    """Write a report to a file.

    Parameters
    ----------
    report : str
        The report to write.
    output_file : str
        The output file.
    """
    with open(output_file, "w") as f:
        f.write(report)


def parse_xyz_file(xyz_content: str) -> List[str]:
    """Parse an XYZ file and return a list of XYZ strings, each representing a molecular structure.

    Parameters
    ----------
    xyz_content : str
        Content of the XYZ file.

    Returns
    -------
    List[str]
        List of XYZ strings, one for each structure found in the file.
    """
    structures = []
    lines = xyz_content.strip().splitlines()
    i = 0

    while i < len(lines):
        try:
            # Get the number of atoms from the first line
            n_atoms = int(lines[i].strip())

            # Check if we have enough lines for this structure
            if i + n_atoms + 1 >= len(lines):
                logging.warning(f"Incomplete structure at position {i}, skipping.")
                break

            # Extract the structure (including the atom count and comment line)
            # This will be n_atoms lines (coordinates) + 2 lines (count and comment) = n_atoms + 2 total lines
            structure_end = i + n_atoms + 2
            structure = "\n".join(lines[i:structure_end])
            structures.append(structure)

            # Move to the next structure
            i = structure_end

        except ValueError:
            # If we can't parse the atom count, skip this line and try the next one
            logging.warning(f"Could not parse atom count at line {i+1}, skipping line.")
            i += 1

        except IndexError:
            # End of file
            break

    if not structures:
        logging.warning("No valid structures found in XYZ content.")

    logging.info(f"Found {len(structures)} structures in XYZ file.")
    return structures


def extract_xyz_comment(xyz_text: str) -> str:
    """Extract the comment line from an XYZ format text.

    Parameters
    ----------
    xyz_text : str
        The XYZ format text.

    Returns
    -------
    str
        The comment line, or an empty string if the comment line cannot be extracted.
    """
    comment = ""
    if not xyz_text or not isinstance(xyz_text, str):
        return comment

    # Normalize line endings to avoid platform-specific issues
    xyz_text = xyz_text.replace("\r\n", "\n").replace("\r", "\n")

    lines = xyz_text.strip().split("\n")
    if len(lines) < 2:
        return comment

    # Try to validate the first line is a number (atom count)
    try:
        atom_count = int(lines[0].strip())
        # The second line is the comment line
        comment = lines[1].strip()
    except (ValueError, IndexError):
        # If the first line isn't a valid integer or there's no second line
        pass

    return comment


def process_file(
    xyz_path, disable_pubchem: bool = False
) -> List[Optional[MoleculeInfo]]:
    """Process a single XYZ file and return a list of MoleculeInfo objects.

    Parameters
    ----------
    xyz_path : str
        Path to the XYZ file
    disable_pubchem : bool, optional
        If True, skip PubChem lookups, by default False.

    Returns
    -------
    List[Optional[MoleculeInfo]]
        List of MoleculeInfo objects, one for each structure in the file.
        If a structure cannot be processed, None will be in its place.
    """
    logging.info(f"Processing {xyz_path}")
    xyz_content = None
    try:
        with open(xyz_path, "r") as f:
            xyz_content = f.read()

        # Parse the XYZ file to get all structures
        structures = parse_xyz_file(xyz_content)

        # If no structures were found, try treating the entire file as a single structure
        if not structures and xyz_content.strip():
            logging.warning(
                f"No structures found with parser, treating file as a single structure."
            )
            structures = [xyz_content]

        molecules = []

        for i, structure_xyz in enumerate(structures):
            # Create molecule from XYZ
            mol = xyz_to_mol(structure_xyz)

            # Extract the comment line
            comment = extract_xyz_comment(structure_xyz)

            # Generate MoleculeInfo
            molecule_info = MoleculeInfo.from_rdkit_mol(
                mol,
                charge=0,
                xyz_data=structure_xyz,
                comment=comment,
                disable_pubchem=disable_pubchem,
            )

            # If mol conversion failed but we have XYZ data
            if (
                mol is None
                and molecule_info is not None
                and molecule_info.xyz_data is not None
            ):
                logging.warning(
                    f"RDKit molecule conversion failed for structure {i+1} in {xyz_path}, but XYZ data is available for 3D view."
                )
                if not molecule_info.formula:
                    # Use filename with index for multiple structures
                    molecule_info.formula = (
                        f"{os.path.basename(xyz_path)} (Structure {i+1})"
                    )
            elif mol is None:
                logging.error(
                    f"Failed to convert structure {i+1} in {xyz_path} to RDKit molecule object."
                )
                molecules.append(None)
                continue

            molecules.append(molecule_info)

        return molecules

    except Exception as e:
        logging.error(f"Error processing {xyz_path}: {e}")
        # If an error occurs but we have xyz_content, we could return a minimal MoleculeInfo for 3D view
        if xyz_content:
            logging.info(
                f"Returning MoleculeInfo with only XYZ data due to processing error for {xyz_path}."
            )
            # Also try to extract the comment
            comment = extract_xyz_comment(xyz_content)
            return [
                MoleculeInfo(
                    xyz_data=xyz_content,
                    formula=os.path.basename(xyz_path),
                    comment=comment,
                )
            ]
        return []


def generate_csv_content(
    molecules: list[MoleculeInfo],
    include_fields: Optional[list[str]] = None,
) -> str:
    """Generate CSV content for a list of molecules.

    Parameters
    ----------
    molecules : list[MoleculeInfo]
        A list of MoleculeInfo objects.
    include_fields : Optional[list[str]], optional
        A list of field names to include. If None, includes default fields:
        index, formula, pubchem_cid, iupac_name, comment, inchi.

    Returns
    -------
    str
        The CSV content as a string.
    """
    import csv
    from io import StringIO

    # Default fields if none specified
    if include_fields is None:
        include_fields = ["formula", "pubchem_cid", "iupac_name", "comment", "inchi"]

    # Validate field names exist in MoleculeInfo
    valid_field_names = {f.name for f in dataclasses.fields(MoleculeInfo)}
    fields_to_include = [f for f in include_fields if f in valid_field_names]

    # Prepare CSV data
    output = StringIO()
    writer = csv.writer(output)

    # Write header row - include index column
    header = ["index"] + fields_to_include
    writer.writerow(header)

    # Write data rows
    for i, molecule in enumerate(molecules):
        row = [i]  # Start with index
        for field in fields_to_include:
            value = getattr(molecule, field, "")
            # Format None values as empty strings
            if value is None:
                value = ""
            row.append(value)
        writer.writerow(row)

    return output.getvalue()


def write_csv(csv_content: str, output_file: str):
    """Write CSV content to a file.

    Parameters
    ----------
    csv_content : str
        The CSV content to write.
    output_file : str
        The output file path.
    """
    with open(output_file, "w") as f:
        f.write(csv_content)
    logging.info(f"CSV file written to {output_file}")


def main():
    """Command line interface for generating reports."""
    import argparse
    import logging
    import os
    import sys
    import tomli

    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Generate HTML reports for molecular structures"
    )
    parser.add_argument(
        "xyz", help="Path to XYZ file or directory containing XYZ files"
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output directory for reports (default: current directory)",
        default=".",
    )
    parser.add_argument(
        "--loglevel",
        help="Set the logging level",
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
    )
    parser.add_argument(
        "--ignore-imag",
        help="Ignore imaginary modes in thermochemistry",
        action="store_true",
    )
    parser.add_argument(
        "--include-props",
        help="Comma-separated list of properties to include in the report (e.g., 'formula,smiles,molecular_weight').",
    )
    parser.add_argument(
        "--exclude-props",
        help="Comma-separated list of properties to exclude from the report (e.g., 'inchi,inchikey').",
    )
    parser.add_argument(
        "--config",
        help="Path to TOML configuration file for report properties",
        default=None,
    )
    parser.add_argument(
        "--view3d",
        help="Enable 3D molecular view in the report using 3Dmol.js.",
        action="store_true",
    )
    parser.add_argument(
        "--view2d",
        help="Enable 2D molecular structure images in the report.",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-view2d",
        help="Disable 2D molecular structure images in the report.",
        action="store_false",
        dest="view2d",
    )
    parser.add_argument(
        "--max-structures",
        help="Maximum number of structures to process per file. Use to avoid memory issues with large files.",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--expanded",
        help="Show all molecular structures expanded by default (may cause performance issues with many molecules).",
        action="store_true",
    )
    parser.add_argument(
        "--disable-pubchem",
        help="Disable PubChem lookups (speeds up report generation).",
        action="store_true",
    )
    parser.add_argument(
        "--csv",
        help="Generate a CSV file in addition to the HTML report.",
        action="store_true",
    )
    parser.add_argument(
        "--csv-only",
        help="Generate only a CSV file without the HTML report.",
        action="store_true",
    )
    parser.add_argument(
        "--csv-fields",
        help="Comma-separated list of fields to include in the CSV file (e.g., 'formula,smiles,comment').",
    )

    # Parse arguments
    args = parser.parse_args()

    # Parse property lists from config file if provided, otherwise use command line
    include_properties = None
    exclude_properties = None
    disable_pubchem = args.disable_pubchem
    generate_csv = args.csv or args.csv_only
    csv_fields = None
    if args.csv_fields:
        csv_fields = [field.strip() for field in args.csv_fields.split(",")]

    if args.config:
        try:
            with open(args.config, "rb") as f:
                config = tomli.load(f)

            if isinstance(config, dict):
                # Get include/exclude properties from config
                if "include_properties" in config:
                    include_properties = config["include_properties"]
                    logging.info(
                        f"Using include_properties from config: {include_properties}"
                    )

                if "exclude_properties" in config:
                    exclude_properties = config["exclude_properties"]
                    logging.info(
                        f"Using exclude_properties from config: {exclude_properties}"
                    )

                # Override 3D view setting if in config
                if "enable_3d_view" in config:
                    args.view3d = config.get("enable_3d_view", args.view3d)
                    logging.info(f"Using 3D view setting from config: {args.view3d}")

                # Override 2D view setting if in config
                if "enable_2d_view" in config:
                    args.view2d = config.get("enable_2d_view", args.view2d)
                    logging.info(f"Using 2D view setting from config: {args.view2d}")

                # Override expanded setting if in config
                if "expanded" in config:
                    args.expanded = config.get("expanded", args.expanded)
                    logging.info(f"Using expanded setting from config: {args.expanded}")

                # Override max structures if in config
                if "max_structures" in config:
                    args.max_structures = config.get(
                        "max_structures", args.max_structures
                    )
                    logging.info(
                        f"Using max_structures from config: {args.max_structures}"
                    )

                # Override disable_pubchem setting if in config
                if "disable_pubchem" in config:
                    disable_pubchem = config.get("disable_pubchem", disable_pubchem)
                    logging.info(
                        f"Using disable_pubchem from config: {disable_pubchem}"
                    )

                # Override CSV settings if in config
                if "generate_csv" in config:
                    generate_csv = config.get("generate_csv", generate_csv)
                    logging.info(f"Using generate_csv from config: {generate_csv}")

                if "csv_fields" in config and not args.csv_fields:
                    csv_fields = config.get("csv_fields")
                    logging.info(f"Using csv_fields from config: {csv_fields}")
        except Exception as e:
            logging.error(f"Error reading config file {args.config}: {e}")
            logging.warning("Using command line arguments instead")

    # Command line arguments override config if both are provided
    if args.include_props:
        include_properties = [p.strip() for p in args.include_props.split(",")]
        logging.info(
            f"Overriding include_properties from command line: {include_properties}"
        )

    if args.exclude_props:
        exclude_properties = [p.strip() for p in args.exclude_props.split(",")]
        logging.info(
            f"Overriding exclude_properties from command line: {exclude_properties}"
        )

    # Set up logging
    log_level = getattr(logging, args.loglevel.upper())
    logging.basicConfig(
        level=log_level, format="IQC %(levelname)s: %(asctime)s - %(message)s"
    )

    if disable_pubchem:
        logging.info("PubChem lookups are disabled")

    if generate_csv:
        logging.info(f"CSV generation is enabled")
        if csv_fields:
            logging.info(f"CSV fields: {csv_fields}")
        else:
            logging.info("Using default CSV fields")

    # Check if input exists
    if not os.path.exists(args.xyz):
        logging.error(f"Input path {args.xyz} does not exist")
        sys.exit(1)

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output):
        os.makedirs(args.output)
        logging.info(f"Created output directory: {args.output}")

    # Process files
    molecules_info_list = []
    report_generated = False
    csv_generated = False

    if os.path.isdir(args.xyz):
        xyz_files = [f for f in os.listdir(args.xyz) if f.endswith(".xyz")]
        if not xyz_files:
            logging.warning(f"No .xyz files found in {args.xyz}")
            sys.exit(0)
        logging.info(
            f"Found {len(xyz_files)} .xyz files to process in directory {args.xyz}"
        )

        for xyz_file in xyz_files:
            input_path = os.path.join(args.xyz, xyz_file)
            molecule_list = process_file(input_path, disable_pubchem=disable_pubchem)

            # Apply max structures limit if specified
            if args.max_structures > 0 and len(molecule_list) > args.max_structures:
                logging.warning(
                    f"Limiting {len(molecule_list)} structures to {args.max_structures} from {xyz_file}"
                )
                molecule_list = molecule_list[: args.max_structures]

            for mol_info in molecule_list:
                if mol_info:
                    molecules_info_list.append(mol_info)

        if molecules_info_list:
            logging.info(
                f"Processed {len(molecules_info_list)} molecules from {len(xyz_files)} files."
            )
            # Generate CSV if requested
            if generate_csv:
                csv_content = generate_csv_content(
                    molecules_info_list, include_fields=csv_fields
                )
                output_filename = "molecules_collection.csv"
                csv_path = os.path.join(args.output, output_filename)
                write_csv(csv_content, csv_path)
                csv_generated = True

            # Generate HTML report if not in CSV-only mode
            if not args.csv_only:
                logging.info(
                    f"Generating combined HTML report for {len(molecules_info_list)} molecules."
                )
                html_report = generate_multi_molecule_html_report(
                    molecules_info_list,
                    include_properties=include_properties,
                    exclude_properties=exclude_properties,
                    enable_2d_view=args.view2d,
                    enable_3d_view=args.view3d,
                    collapsed_by_default=not args.expanded,
                )
                output_filename = "molecules_collection_report.html"
                output_path = os.path.join(args.output, output_filename)
                write_report(html_report, output_path)
                logging.info(f"Combined report written to {output_path}")
                report_generated = True
        else:
            logging.warning("No molecules could be processed from the directory.")

    else:  # Single file
        molecule_list = process_file(args.xyz, disable_pubchem=disable_pubchem)

        # Apply max structures limit if specified
        if args.max_structures > 0 and len(molecule_list) > args.max_structures:
            logging.warning(
                f"Limiting {len(molecule_list)} structures to {args.max_structures} from {args.xyz}"
            )
            molecule_list = molecule_list[: args.max_structures]

        base_name = os.path.splitext(os.path.basename(args.xyz))[0]

        # If we have more than one molecule in the file, treat it as a multi-molecule file
        if len(molecule_list) > 1:
            logging.info(f"Found {len(molecule_list)} structures in file {args.xyz}")
            valid_molecules = [mol for mol in molecule_list if mol is not None]

            if valid_molecules:
                molecules_info_list.extend(valid_molecules)

                # Generate CSV if requested
                if generate_csv:
                    csv_content = generate_csv_content(
                        valid_molecules, include_fields=csv_fields
                    )
                    output_filename = f"{base_name}.csv"
                    csv_path = os.path.join(args.output, output_filename)
                    write_csv(csv_content, csv_path)
                    csv_generated = True

                # Generate HTML report if not in CSV-only mode
                if not args.csv_only:
                    logging.info(f"Generating multi-structure report for {args.xyz}")
                    html_report = generate_multi_molecule_html_report(
                        valid_molecules,
                        include_properties=include_properties,
                        exclude_properties=exclude_properties,
                        enable_2d_view=args.view2d,
                        enable_3d_view=args.view3d,
                        collapsed_by_default=not args.expanded,
                    )
                    output_filename = f"{base_name}_structures.html"
                    output_path = os.path.join(args.output, output_filename)
                    write_report(html_report, output_path)
                    logging.info(f"Multi-structure report written to {output_path}")
                    report_generated = True

            else:
                logging.error(
                    f"None of the {len(molecule_list)} structures in {args.xyz} could be processed."
                )

        # If only one structure, proceed as before
        elif len(molecule_list) == 1:
            mol_info = molecule_list[0]
            if mol_info:
                molecules_info_list.append(mol_info)

                # Generate CSV if requested
                if generate_csv:
                    csv_content = generate_csv_content(
                        [mol_info], include_fields=csv_fields
                    )
                    output_filename = f"{base_name}.csv"
                    csv_path = os.path.join(args.output, output_filename)
                    write_csv(csv_content, csv_path)
                    csv_generated = True

                # Generate HTML report if not in CSV-only mode
                if not args.csv_only:
                    logging.info(f"Generating report for single molecule: {args.xyz}")
                    html_report = generate_multi_molecule_html_report(
                        [mol_info],
                        include_properties=include_properties,
                        exclude_properties=exclude_properties,
                        enable_2d_view=args.view2d,
                        enable_3d_view=args.view3d,
                        collapsed_by_default=False,  # Always expanded for single molecule
                    )
                    output_filename = f"{base_name}.html"
                    output_path = os.path.join(args.output, output_filename)
                    write_report(html_report, output_path)
                    logging.info(f"Report written to {output_path}")
                    report_generated = True

            else:
                logging.error(f"Could not process the single file: {args.xyz}")
        else:
            logging.error(f"No valid structures found in file: {args.xyz}")

    if report_generated or csv_generated:
        logging.info("Report generation complete.")
    else:
        logging.error("No reports or CSV files were generated.")
        sys.exit(1)


if __name__ == "__main__":
    main()
