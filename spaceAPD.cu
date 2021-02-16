#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "typeDefinition.cuh"
//#include "globalVariables.cuh"
#include "hostPrototypes.h"
#include "devicePrototypes.cuh"

#include "./common/CudaSafeCall.h"

extern __constant__ int nx_d, ny_d;
extern __device__ int contour_count;
extern __constant__ REAL dt_d, hx_d, hy_d;
extern paramVar param;

__global__ void stim_kernel(REAL *u, REAL *stimulus) {

  unsigned int index = threadIdx.x + blockDim.x*blockIdx.x;
  unsigned int stride = blockDim.x*gridDim.x;

  while (index < nx_d*ny_d) {

    u[index] += stimulus[index];

    index += stride;

  }

}

void stim_wrapper(dim3 grid1D, dim3 block1D, REAL *u, REAL *stimulus) {

  stim_kernel<<<grid1D,block1D>>>(u,stimulus);
  CudaCheckError();

}

__global__ void countour_kernel(size_t pitch, REAL *sAPD, 
  bool *sAPD_plot, bool *stimArea, int *contour_count, float3 *contour_vector,
  float physicalTime) {

  /*------------------------------------------------------------------------
  * getting i and j global indices
  *------------------------------------------------------------------------
  */

  const int i = blockIdx.x*BLOCK_DIM_X + threadIdx.x;
  const int j = blockIdx.y*BLOCK_DIM_Y + threadIdx.y;

  if ( (i<nx_d) && (j<ny_d) ) {

  /*------------------------------------------------------------------------
  * converting global index into matrix indices assuming
  * the column major structure of the matlab matrices
  *------------------------------------------------------------------------
  */

  const int i2d = I2D(nx_d,i,j);

  REAL v0, v1x, v1y, ppx, ppy , zpmx, zpmy;
  float t;
  float3 pxyt;

  // Avoid region near pacing site
  bool sc = stimArea[i2d];

  /*------------------------------------------------------------------------
  * Find contour points
  *------------------------------------------------------------------------
  */

  // sAPD[i2d] = (i>0) && (j>0) && (i<(nx_d-1)) && (j<(ny_d-1)) ? sAPD[i2d] : 1.0;

  v0 = sAPD[i2d];
  v1x = sAPD[I2D(nx_d,i+1,j)];
  v1y = sAPD[I2D(nx_d,i,j+1)];

  zpmx = i < (nx_d-1) ? v0*v1x : v0*v0;
  zpmy = j < (ny_d-1) ? v0*v1y : v0*v0;

  // if ((i == nx_d/2) && (j == ny_d/2)) {
  //   printf("zpmx = %f\n",zpmx);
  // }

	if ( (zpmx < 0.0) || (zpmy < 0.0) && sc ) {

    ppx = abs(v0-v1x) > MACHINE_EPS ? i + v0/(v0-v1x) : i ;
    ppy = abs(v0-v1y) > MACHINE_EPS ? j + v0/(v0-v1y) : j ;
    t = physicalTime;

    pxyt = make_float3( (float)ppx, (float)ppy, t );
    #ifdef SAVE_DATA
		  push_back3(pxyt,contour_count,contour_vector);
    #endif
    plot_field(pxyt.x,pxyt.y,sAPD_plot);

	}

}

}

__global__ void countourNewton_kernel(size_t pitch, REAL *subAPD, REAL *divAPD,
  bool *sAPD_plot, bool *stimArea, int *contour_count, float3 *contour_vector,
  float physicalTime) {

/*------------------------------------------------------------------------
  * Getting i, and k global indices
  *------------------------------------------------------------------------
  */

  const int i = blockIdx.x*BLOCK_DIM_X + threadIdx.x;
  const int j = blockIdx.y*BLOCK_DIM_Y + threadIdx.y;

  if ( (i<nx_d) && (j<ny_d) ) {

  int s0   = I2D(nx_d,i,j);
  int sx   = ( i<(nx_d-1)  ) ? I2D(nx_d,i+1,j) : I2D(nx_d,i,j);
  int sy   = ( j<(ny_d-1)  ) ? I2D(nx_d,i,j+1) : I2D(nx_d,i,j);
  int sxy  = ( (j<(ny_d-1)) && (i<(nx_d-1) ) ) ? I2D(nx_d,i+1,j+1) : I2D(nx_d,i,j);

  REAL px1, px2, px3, px4, py1, py2, py3, py4;

  /*------------------------------------------------------------------------
  * XY plane
  *------------------------------------------------------------------------
  */

  px1 = subAPD[s0];
  px2 = subAPD[sx];
  px4 = subAPD[sy];
  px3 = subAPD[sxy];

  py1 = divAPD[s0];
  py2 = divAPD[sx];
  py4 = divAPD[sy];
  py3 = divAPD[sxy];

  REAL r1, r2, J11, J12, J21, J22, detJ, u1, u2;
  REAL s = 0.5;
  REAL t = 0.5;
  int maxIter = 10;
  float3 pxyt;

  int k;

  /*------------------------------------------------------------------------
  * newton method
  *------------------------------------------------------------------------
  */

  for (k=0;k<maxIter;k++) {

    r1 = px1*(1.0-s)*(1.0-t) + px2*s*(1.0-t) + px3*s*t + px4*(1.0-s)*t; //residual
    r2 = py1*(1.0-s)*(1.0-t) + py2*s*(1.0-t) + py3*s*t + py4*(1.0-s)*t; //residual
    r1 = r1 - r2;
    r2 = r2 - r1;
    // r1 = px1*(1.0-s)*(1.0-t) + px2*s*(1.0-t) + px3*s*t + px4*(1.0-s)*t - 0.0; //residual
    // r2 = py1*(1.0-s)*(1.0-t) + py2*s*(1.0-t) + py3*s*t + py4*(1.0-s)*t - 1.0; //residual
    // J11 = -px1*(1.0-t) + px2*(1.0-t) + px3*t - px4*t; // dr1/ds
    // J21 = -py1*(1.0-t) + py2*(1.0-t) + py3*t - py4*t; // dr2/ds
    // J12 = -px1*(1.0-s) - px2*s + px3*s + px4*(1.0-s); // dr1/dt
    // J22 = -py1*(1.0-s) - py2*s + py3*s + py4*(1.0-s); // dr2/dt

    J11 = -px1*(1.0-t) + px2*(1.0-t) + px3*t - px4*t -(-py1*(1.0-t) + py2*(1.0-t) + py3*t - py4*t); // dr1/ds
    J21 = -py1*(1.0-t) + py2*(1.0-t) + py3*t - py4*t -(-px1*(1.0-t) + px2*(1.0-t) + px3*t - px4*t); // dr2/ds
    J12 = -px1*(1.0-s) - px2*s + px3*s + px4*(1.0-s) -(-py1*(1.0-s) - py2*s + py3*s + py4*(1.0-s)); // dr1/dt
    J22 = -py1*(1.0-s) - py2*s + py3*s + py4*(1.0-s) -(-px1*(1.0-s) - px2*s + px3*s + px4*(1.0-s)); // dr2/dt

    detJ = J11*J22 - J12*J21;

    if ( !equals( detJ, 0.0, 1e-14 ) ) {

      s -= (J22*r1 - J12*r2)/detJ;
      t -= (-J21*r1 + J11*r2)/detJ;
      s = min(max(s, 0.0),1.0);
      t = min(max(t, 0.0),1.0);

    } else {
      s = -1.0; // Arbitrary values
      t = -1.0;
    }

  }

  // Set the negative condition to a value you are not looking for.
  u1 = (s>=0.0) && (s<=1.0) && (t>=0.0) && (t<=1.0) ? 
    px1*(1.0-s)*(1.0-t) + px2*s*(1.0-t) + px3*s*t + px4*(1.0-s)*t : 1.0;
  u2 = (s>=0.0) && (s<=1.0) && (t>=0.0) && (t<=1.0) ? 
    py1*(1.0-s)*(1.0-t) + py2*s*(1.0-t) + py3*s*t + py4*(1.0-s)*t : 0.0;

  // if ( !equals( u1, 0.0, 1e-15 ) && !equals( u2, 0.0, 1e-15 ) ) {
  if ( equals( u1, u2, 1e-15 ) ) {

    pxyt = make_float3( (float)i+(float)s, (float)j+(float)t, (float)physicalTime );
    #ifdef SAVE_DATA
      push_back3(pxyt,contour_count,contour_vector);
    #endif
    plot_field(pxyt.x,pxyt.y,sAPD_plot);

  }

}

}

// }

void countour_wrapper(size_t pitch, dim3 grid2D, dim3 block2D,
	REAL *sAPD, REAL *dAPD, bool *sAPD_plot, bool *stimArea, int *contour_count, 
  float3 *contour_vector, float physicalTime) {

  CudaSafeCall(cudaMemset(contour_count,0,sizeof(int))); // Initialize number of contour points
  CudaSafeCall(cudaMemset(sAPD_plot,0,param.nx*param.ny*sizeof(bool))); // Reset screen contour plot

	countour_kernel<<<grid2D,block2D>>>(pitch,sAPD,sAPD_plot,stimArea,
    contour_count,contour_vector,physicalTime);
	CudaCheckError();

  // countourNewton_kernel<<<grid2D,block2D>>>(pitch,sAPD,dAPD,sAPD_plot,stimArea,
  //   contour_count,contour_vector,physicalTime);
  // CudaCheckError();

}

__global__ void sAPD_kernel(size_t pitch, int count, REAL *uold, REAL *unew,
  REAL *APD1, REAL *APD2, REAL *sAPD, REAL *dAPD, REAL *back, REAL *front, bool *first, 
  bool *stimArea, bool stimulate) {

  unsigned int index = threadIdx.x + blockDim.x*blockIdx.x;
  unsigned int stride = blockDim.x*gridDim.x;

  REAL uo, un, apdTh;
  bool sc;

  apdTh = 0.15;

  while (index < nx_d*ny_d) {

    uo = uold[index];
    un = unew[index];
    sc = stimArea[index];

    if (stimulate) {  

      // Front
      if ( (un > apdTh) && (uo < apdTh) && sc ) {
        front[index] = dt_d*(count - (un - apdTh)/(un - uo));
      }

      // Back
      if ( (un < apdTh) && (uo > apdTh) &&  sc ) {
        back[index] = dt_d*(count - (un - apdTh)/(un - uo));
      }
      
      // if (index==(nx_d*ny_d/2+nx_d/2)) {
      //   printf("%f\t%f\t%f\n", un, uo,sAPD[index]);
      // }

      if ( (back[index] > 0.0) && (front[index] > 0.0) && (first[index]==false) && sc ) {
        APD1[index] = back[index]-front[index];
        front[index] = 0.0;
        back[index] = 0.0;
        first[index] = true;
        }

      if ( (back[index] > 0.0) && (front[index] > 0.0) && first[index] && sc ) {
        APD2[index] = back[index]-front[index];
        front[index] = 0.0;
        back[index] = 0.0;
        first[index] = false;
        }

        // if (index == (nx_d*ny_d/2 + nx_d/2)) {
        //   printf("APD = %f\n", APD1[index]-APD2[index]);
        // }

       // Translate the APD contours (nodes) to the zero-surface
        sAPD[index] = (APD1[index]-APD2[index]> 0.0) && sc ? 1.0 : -1.0;
        // sAPD[index] =  APD1[index];
        // APD1[index]-APD2[index];//(APD1[index]-APD2[index]> 0.0) && sc ? 1.0 : -1.0;
        sAPD[index] *= (REAL)sc;
        dAPD[index] =  APD2[index];
        //abs(APD2[index])>0.0 ? APD1[index]/APD2[index] : 0.0; //(APD1[index]-APD2[index]> 0.0) && sc ? 1.0 : -1.0;
        dAPD[index] *= (REAL)sc;


    } else {

    // Front
      if ( (un > apdTh) && (uo < apdTh) ) {
        front[index] = dt_d*(count - (un - apdTh)/(un - uo));
      }

      // Back
      if ( (un < apdTh) && (uo > apdTh) ) {
        back[index] = dt_d*(count - (un - apdTh)/(un - uo));
      }

      if ( (back[index] > 0.0) && (front[index] > 0.0) && (first[index]==false) ) {
        APD1[index] = back[index]-front[index];
        front[index] = 0.0;
        back[index] = 0.0;
        first[index] = true;
        }

      if ( (back[index] > 0.0) && (front[index] > 0.0) && first[index] ) {
        APD2[index] = back[index]-front[index];
        front[index] = 0.0;
        back[index] = 0.0;
        first[index] = false;
        }

      sAPD[index] = (APD1[index]-APD2[index]> 0.0) ? 1.0 : -1.0;
      // dAPD[index] = APD1[index]/APD2[index];
    }

    index += stride;

  }

}

void sAPD_wrapper(size_t pitch, dim3 grid1D, dim3 block1D, int count, 
  REAL *uold, REAL *unew, REAL *APD1, REAL *APD2, REAL *sAPD, REAL *dAPD, 
  REAL *back, REAL *front, bool *first, bool *stimArea, bool stimulate) {

  sAPD_kernel<<<grid1D,block1D>>>(pitch,count,uold,unew,
    APD1,APD2,sAPD,dAPD,back,front,first,stimArea,stimulate);
  CudaCheckError();

}