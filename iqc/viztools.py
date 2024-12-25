import ipywidgets as widgets
from IPython.display import display, HTML
import nglview as nv
import ase.io
from io import StringIO
import requests
import tarfile
from io import BytesIO
import json


def display_filtered_data(df, display_columns):
    """
    Displays a filtered view of the given DataFrame using interactive widgets.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data to be displayed.
    display_columns (list): A list of column names to be displayed.

    This function creates interactive widgets for filtering the DataFrame based on
    selected columns and values. It also provides an index selector to navigate through
    the filtered data. The filtered data and corresponding visualizations are displayed
    using output widgets.
    """
    import ipywidgets as widgets
    from IPython.display import display, HTML
    import nglview as nv
    import ase.io
    from io import StringIO

    # Create filter widgets
    filter_keys = [
        widgets.Dropdown(
            options=["none"] + display_columns,
            description=f"Filter {i+1} by:",
            value="none",
        )
        for i in range(3)
    ]

    filter_values = [
        widgets.Text(description=f"Value {i+1}:", disabled=True) for i in range(3)
    ]

    # Create index selector
    index_selector = widgets.IntText(
        value=0,
        min=0,
        max=len(df) - 1,
        description="Index:",
        step=1,
    )

    # Create output widgets
    info_output = widgets.Output()
    view_output = widgets.Output()

    def update_filter_values(*args):
        for i in range(3):
            filter_values[i].disabled = filter_keys[i].value == "none"

    def update_display(*args):
        info_output.clear_output()
        view_output.clear_output()

        with info_output:
            # Apply filters if selected
            filtered_df = df.copy()
            for i in range(3):
                if filter_keys[i].value != "none" and filter_values[i].value:
                    try:
                        column = filter_keys[i].value
                        value = filter_values[i].value
                        if filtered_df[column].dtype in ["int64", "float64"]:
                            filtered_df = filtered_df[
                                filtered_df[column] == float(value)
                            ]
                        else:
                            filtered_df = filtered_df[
                                filtered_df[column]
                                .astype(str)
                                .str.contains(value, case=False)
                            ]
                    except ValueError:
                        print(f"Invalid filter value for Filter {i+1}: {value}")
                        return
                    except KeyError:
                        print(f"Invalid column for Filter {i+1}: {column}")
                        return

            if len(filtered_df) == 0:
                print("No matching records")
                return

            # Update slider max value
            index_selector.max = len(filtered_df) - 1

            # Display selected row
            selected_row = filtered_df.iloc[index_selector.value]
            print(f"Showing record {index_selector.value} of {len(filtered_df)-1}")
            print("\nName:", selected_row["unique_name"].split("202412")[0])
            print("-" * 50)
            for col in display_columns:
                if col in selected_row:
                    print(f"{col}: {selected_row[col]}")

        with view_output:
            try:
                initial_atoms = ase.io.read(
                    StringIO(selected_row["initial_xyz"]), format="xyz"
                )
                optimized_atoms = ase.io.read(
                    StringIO(selected_row["opt_xyz"]), format="xyz"
                )
                # Create NGLView widgets
                view2 = nv.show_ase(initial_atoms)
                view1 = nv.show_ase(optimized_atoms)

                # Set display options
                for view in [view1, view2]:
                    view.center()
                    view.focus()
                    view.camera = "orthographic"
                    view.parameters = {
                        "clipDist": -50,
                        "backgroundColor": "white",
                    }

                # Create labels
                label2 = widgets.Label("Initial Structure")
                label1 = widgets.Label("Optimized Structure")

                # Arrange views
                display(widgets.VBox([widgets.VBox([label1, view1, label2, view2])]))

            except Exception as e:
                print(f"Error displaying structures: {str(e)}")

    # Connect callbacks
    for i in range(3):
        filter_keys[i].observe(update_filter_values, "value")
        filter_values[i].observe(update_display, "value")
        filter_keys[i].observe(update_display, "value")
    index_selector.observe(update_display, "value")

    # Create layout
    filters = widgets.HBox(
        [widgets.VBox([filter_keys[i], filter_values[i]]) for i in range(3)]
    )
    widget_box = widgets.VBox(
        [filters, index_selector, widgets.VBox([info_output, view_output])]
    )

    # Display initial state
    display(widget_box)
    update_display()


def read_json_from_tar_gz(url):
    """
    Reads a tar.gz file from a URL, extracts JSON files, and returns a list of JSON objects.
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes

        with tarfile.open(fileobj=BytesIO(response.content), mode="r:gz") as tar:
            json_data = []
            for member in tar.getmembers():
                if member.isfile() and member.name.endswith(".json"):
                    file_content = tar.extractfile(member).read()
                    try:
                        json_object = json.loads(file_content)
                        json_data.append(json_object)
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON in file {member.name}: {e}")
            return json_data
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        return None
    except tarfile.TarError as e:
        print(f"Error reading tar file: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None
