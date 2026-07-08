#!/bin/bash -l
#---------------- PBS options ----------------#
#PBS -N iqc_thermo
#PBS -l select=8:system=crux
#PBS -l walltime=24:00:00
#PBS -q workq
#PBS -A IQC
#PBS -l filesystems=home:grand:eagle
#---------------------------------------------#

################## Environment ################
#----------- Micromamba environment ----------#
eval "$(micromamba shell hook -s bash)"
micromamba activate iqc-env

################ Job layout ###################
NNODES=$(wc -l < "$PBS_NODEFILE")

RANKS_PER_NODE=4        #  MPI rank per node
NTHREADS=32          # one core per thread (dual-socket EPYC 7742 gives 128 physical cores)
DEPTH=$NTHREADS      # PALS reservation must match threads used

export OMP_NUM_THREADS=$NTHREADS
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

NTOTRANKS=$(( NNODES * RANKS_PER_NODE ))

################ Run command ##################
cd "$PBS_O_WORKDIR"
XYZDIR="/grand/projects/IQC/data/photocarboxylation/structures/metal-bipy-alkyne_structures"
XYZ="/lus/grand/projects/IQC/data/photocarboxylation/structures/refined_consumer_available/filtered_structures_6256.xyz"
XYZ="/lus/grand/projects/IQC/keceli/ASA/IQC/run/rosmi/rosmi_xyz"
LOG="thermo_n${NNODES}_r${NTOTRANKS}_t${NTHREADS}.log"
echo "Nodes=$NNODES  Ranks/node=$RANKS_PER_NODE TotalRanks=$NTOTRANKS  OMP=$NTHREADS"   | tee  "$LOG"
echo "Started: $(date '+%F %T')"                                      | tee -a "$LOG"

MPI_ARGS="-n ${NTOTRANKS} --ppn ${RANKS_PER_NODE} --depth=${DEPTH} --cpu-bind depth "
OMP_ARGS="--env OMP_NUM_THREADS=${NTHREADS} --env OMP_PROC_BIND=true --env OMP_PLACES=cores "

#/lus/eagle/projects/datascience/keceli/conda_envs/iqc-env/bin/mpiexec -n $NTOTRANKS --oversubscribe --bind-to core --map-by core iqc -x "${XYZDIR}/test.xyz" -t thermo -l DEBUG 2>&1 | tee -a "$LOG"
/opt/cray/pals/1.6/bin/mpiexec ${MPI_ARGS} ${OMP_ARGS}  iqc -x $XYZ -t thermo --ignore-imag -l DEBUG 2>&1 | tee -a "$LOG"

echo "Finished: $(date '+%F %T')" | tee -a "$LOG"
################################################

