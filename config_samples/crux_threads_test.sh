#!/bin/bash -l
#PBS -l select=1:system=crux
#PBS -l walltime=1:59:00
#PBS -q workq
#PBS -A Catalyst
#PBS -l filesystems=home:grand:eagle

#–– Node & rank layout
NNODES=$(wc -l < $PBS_NODEFILE)
NRANKS_PER_NODE=1                   # one MPI rank per node
NTOTRANKS=$(( NNODES * NRANKS_PER_NODE ))
NDEPTH=256                          # bind over all 256 HW threads

#–– Paths & env
XYZDIR="/home/keceli/IQC/data/Ni_bipy-all-q-C2H2-p"
cd $PBS_O_WORKDIR
micromamba activate iqc-env

#–– Master log header
MASTER_LOG=thread_scaling.log
echo "Nodes: $NNODES, Ranks/node: $NRANKS_PER_NODE, Depth: $NDEPTH" > $MASTER_LOG
echo "Testing OMP thread counts on $(date '+%Y-%m-%d %H:%M:%S')" >> $MASTER_LOG
echo "" >> $MASTER_LOG

#–– Sweep over thread counts
for NTHREADS in 1 2 4 8 16 32 64 128 256; do
  export OMP_NUM_THREADS=$NTHREADS
  RUN_LOG=thread_${NTHREADS}.log

  echo "=== OMP_NUM_THREADS=${NTHREADS} ===" | tee -a $MASTER_LOG > $RUN_LOG

  start_ts=$(date '+%Y-%m-%d %H:%M:%S')
  start_s=$(date +%s)
  echo "Start: $start_ts" | tee -a $MASTER_LOG $RUN_LOG

  /opt/cray/pals/1.3.4/bin/mpiexec \
    -n $NTOTRANKS \
    --ppn $NRANKS_PER_NODE \
    --depth=$NDEPTH \
    --cpu-bind=depth \
    --env OMP_NUM_THREADS=$OMP_NUM_THREADS \
    --env OMP_PROC_BIND=spread \
    --env OMP_PLACES=threads \
    ${CONDA_PREFIX}/bin/iqc -t opt -x ${XYZDIR}/Ni_bipy-nacaaaaa_q-C2H2-p.xyz -p config.yaml \
    2>&1 | tee -a $MASTER_LOG $RUN_LOG

  end_ts=$(date '+%Y-%m-%d %H:%M:%S')
  end_s=$(date +%s)
  elapsed=$(( end_s - start_s ))

  echo "End:   $end_ts" | tee -a $MASTER_LOG $RUN_LOG
  echo "Elapsed: ${elapsed} seconds" | tee -a $MASTER_LOG $RUN_LOG
  echo "" | tee -a $MASTER_LOG $RUN_LOG
done

echo "All tests complete at $(date '+%Y-%m-%d %H:%M:%S')" | tee -a $MASTER_LOG
