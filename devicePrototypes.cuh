

__global__ void reactionDiffusion_kernel(size_t pitch,
  stateVar g_out, stateVar g_in,
  stateVar velTan, bool reduceSym, bool *solid);
__global__ void spiralTip_kernel(size_t pitch, REAL *g_past,
  REAL *g_present, bool *tip_plot, int *tip_count, vec5dyn *tip_vector,
  REAL physicalTime);
__global__ void spiralTipNewton_kernel(size_t pitch, REAL *g_past,
  REAL *g_present, bool *tip_plot, int *tip_count, vec5dyn *tip_vector,
  REAL physicalTime);
__device__ void tipRecordPlane(const int i, const int j, float2 tip, REAL disc,
  REAL pTime, REAL *g_p, bool *tip_plot, int *tip_count, vec5dyn *tip_vector);
__device__ float2 gradient(const int i, const int j, REAL s, REAL t, REAL *g_p);
__device__ bool abubuFilament(int s0, int sx, int sy,  int sxy, 
    REAL *g_past, REAL *g_present);

__global__ void advFDBFECC_kernel(size_t pitch, stateVar g_out, stateVar g_in,  
  stateVar conv, stateVar uf, stateVar ub, stateVar ue, bool *solid);
__global__ void advSLBFECC_kernel(size_t pitch, stateVar gOut, stateVar gIn,  
  stateVar conv, REAL *uf, REAL *ub, REAL *ue, bool *solid);
__global__ void slice_kernel(stateVar g, sliceVar slice, sliceVar slice0,
  bool reduceSym, bool reduceSymStart, stateVar conv, int scheme,
  bool *intglArea, int *tip_count, vec5dyn *tip_vector);
__global__ void trapz_kernel(REAL *f, REAL *g, REAL *h, REAL *w, 
  REAL *dot, REAL *coeffTrapz, int *tip_count, vec5dyn *tip_vector,
  int count);
__global__ void Cxy_field_kernel(stateVar conv, REAL3 c, REAL3 phi, bool *solid);
void __global__ singleCell_kernel(size_t pitch, stateVar g_out, 
  REAL *pt_d, int2 point);

__global__ void stim_kernel(REAL *u, REAL *stimulus);
__global__ void countour_kernel(size_t pitch, REAL *sAPD, REAL *dAPD,
  bool *sAPD_plot, bool *stimArea, int *contour_count, float3 *contour_vector,
  float physicalTime);
__global__ void sAPD_kernel(size_t pitch, int count, REAL *uold, REAL *unew,
  REAL *APD1, REAL *APD2, REAL *sAPD, REAL *dAPD, REAL *back, REAL *front, bool *first,
  bool *stimArea, bool stimulate);
__global__ void countourNewton_kernel(size_t pitch, REAL *subAPD, REAL *divAPD,
  bool *sAPD_plot, bool *stimArea, int *contour_count, float3 *contour_vector,
  float physicalTime);


__global__ void get_rgba_kernel (size_t pitch, int ncol,
                                 REAL *field,
                                 unsigned int *plot_rgba_data,
                                 unsigned int *cmap_rgba_data,
                                 bool *lines);

__device__ void push_back3(float3 pt, int *count, float3 *count_vector);
__device__ void push_back5(vec5dyn pt, int *count, vec5dyn *count_vector);
__device__ void plot_field(float ptx, float pty, bool *plot_array);
__device__ bool equals( REAL a, REAL b, REAL tolerance );
__device__ int coord_i(int i);
__device__ int coord_j(int j);
__device__ int sign(REAL x);
__host__ __device__ int iDivUp(int a, int b);
__device__ inline REAL my_lerp(REAL v0, REAL v1, REAL t);

__device__ REAL convCentralX(REAL *f, int E, int W);
__device__ REAL convCentralY(REAL *f, int N, int S);
__device__ REAL convCentral2X(REAL *f, const int i, const int j, int E, int W);
__device__ REAL convCentral2Y(REAL *f, const int i, const int j, int N, int S);
__device__ REAL convFB2ndOX(REAL *f, const int i, const int j, 
  int C, int E, int W, stateVar conv);
__device__ REAL convFB2ndOY(REAL *f, const int i, const int j, 
  int C, int N, int S, stateVar conv);
// __device__ REAL MUSCLx(REAL *f, const int C, int W, int E);
// __device__ REAL MUSCLy(REAL *f, const int C, int S, int N);
__device__ REAL MUSCLx(REAL *f, const int i, const int j, const int C, int W, int E);
__device__ REAL MUSCLy(REAL *f, const int i, const int j, const int C, int S, int N);
