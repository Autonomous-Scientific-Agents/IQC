# IQC
Interactive Quantum Chemistry


# Troubleshooting
If you see (mostly on Mac with OpenMPI):
```
shmem: mmap: an error occurred while determining whether or not /tmp/ompi.yv.1001/jf.0/3074883584/sm_segment.yv.1001.b7470000.0 could be created
``
Try: `export OMPI_MCA_btl_sm_backing_directory=/tmp`
