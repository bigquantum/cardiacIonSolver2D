
#include <stdio.h>
#include <stdlib.h>

#include "typeDefinition.cuh"
//#include "globalVariables.cuh"
#include "hostPrototypes.h"
#include "devicePrototypes.cuh"

#include "./common/CudaSafeCall.h"

extern __constant__ int nx_d, ny_d;
extern __constant__ REAL dt_d, rx_d, ry_d;
extern __constant__ REAL rxy_d, rbx_d, rby_d, rscale_d; 
extern __constant__ REAL tau_e_d, tau_n_d, e_h_d, e_n_d, e_star_d, Re_d, M_d_d, expRe_d, tc_d;
extern __constant__ REAL boundaryVal_d;
extern __constant__ bool circle_d, neumannBC_d, gateDiff_d, anisotropy_d;

/*========================================================================
 * Main Entry of the Kernel
 *========================================================================
*/

__global__ void reactionDiffusion_kernel(size_t pitch,
  stateVar g_out, stateVar g_in,
  stateVar velTan, bool reduceSym, bool *solid) {

  /*------------------------------------------------------------------------
  * getting i and j global indices
  *-------------------------------------------------------------------------
  */

  const int i = blockIdx.x*blockDim.x + threadIdx.x;
  const int j = blockIdx.y*blockDim.y + threadIdx.y;

  if ( (i<nx_d) && (j<ny_d) ) {

  /*------------------------------------------------------------------------
  * converting global index into matrix indices assuming
  * the column major structure of the matlab matrices
  *-------------------------------------------------------------------------
  */

  const int i2d = i + j*nx_d;

  /*------------------------------------------------------------------------
  * setting local variables
  *-------------------------------------------------------------------------
  */

  REAL du2dt, dv2dt;
  REAL u = g_in.u[i2d] ;
  REAL v = g_in.v[i2d] ;

  /*------------------------------------------------------------------------
  * Additional heaviside functions
  *-------------------------------------------------------------------------
  */

  REAL p = ( u > e_n_d  ) ? 1.0 : 0.0 ;

  /*------------------------------------------------------------------------
  * I_sum (voltage)
  *-------------------------------------------------------------------------
  */

  REAL I_sum = -(-u + (e_star_d-pow(v,M_d_d))*(1.0 - tanh(u-e_h_d))*0.5*u*u)/tau_e_d;

  /*------------------------------------------------------------------------
  * Calculating the reaction for the gates
  *------------------------------------------------------------------------
  */

  REAL I_v = -(((1.0 - (1.0 - expRe_d) * v)/(1.0 - expRe_d))*p - (1.0 - p)*v)/tau_n_d;

  /*------------------------------------------------------------------------
  * Laplacian Calculation
  *
  * No flux boundary condition is applied on all boundaries through
  * the Laplacian operator definition
  *------------------------------------------------------------------------
  */

  /////////////////////////////
  // Solve homogeneous tissue
  /////////////////////////////

  if ( neumannBC_d ) {

    int S = I2D(nx_d,i,coord_j(j-1));
    int N = I2D(nx_d,i,coord_j(j+1));
    int W = I2D(nx_d,coord_i(i-1),j);
    int E = I2D(nx_d,coord_i(i+1),j);

    if ( circle_d ) {

      bool sc = solid[i2d];
      bool sw = solid[W];
      bool se = solid[E];
      bool sn = solid[N];
      bool ss = solid[S];
        
      float3 coeffx =
      make_float3( (sw && se) && (sw && sc) ? 1.0 : ( (sw && sc) ? 2.0 : 0.0) ,
                    sc ? ( (sw || se) ? 2.0 : 0.0 ) : 0.0 ,
                    (sw && se) && (sc && se) ? 1.0 : ( (sc && se) ? 2.0 : 0.0 ));
      float3 coeffy =
      make_float3( (sn && ss) && (sn && sc) ? 1.0 : ( (sn && sc) ? 2.0 : 0.0 ) ,
                    sc ? ( (sn || ss) ? 2.0 : 0.0 ) : 0.0 ,
                    (sn && ss) && (sc && ss) ? 1.0 : ( (sc && ss) ? 2.0 : 0.0 ));

      du2dt = (
           ( coeffx.x*g_in.u[W] - coeffx.y*u + coeffx.z*g_in.u[E] )*rx_d
      +    ( coeffy.x*g_in.u[N] - coeffy.y*u + coeffy.z*g_in.u[S] )*ry_d );

      dv2dt = 0.0;

      if ( gateDiff_d ) {

        // Gate 1
        dv2dt = (
             ( coeffx.x*g_in.v[W] - coeffx.y*v + coeffx.z*g_in.v[E] )*rx_d*rscale_d
        +    ( coeffy.x*g_in.v[N] - coeffy.y*v + coeffy.z*g_in.v[S] )*ry_d*rscale_d );

      }

      // Anisotropic mode pending FIXME

    } else { // Square boundary

      du2dt = (
         ( g_in.u[W] - 2.0*u + g_in.u[E] )*rx_d
      +  ( g_in.u[N] - 2.0*u + g_in.u[S] )*ry_d );

      dv2dt = 0.0;

      if ( gateDiff_d ) {

        // Gate 1
        dv2dt = (
           ( g_in.v[W] - 2.0*v + g_in.v[E] )*rx_d*rscale_d
        +  ( g_in.v[N] - 2.0*v + g_in.v[S] )*ry_d*rscale_d );

      }

      if ( anisotropy_d ) {

        int SWxy = (i>0  && j>0) ? I2D(nx_d,i-1,j-1) :
                  ((i==0 && j>0) ? I2D(nx_d,i+1,j-1) :
                  ((i>0  && j==0)? I2D(nx_d,i-1,j+1) : I2D(nx_d,i+1,j+1) ) ) ;

        int SExy = (i<(nx_d-1)  && j>0) ? I2D(nx_d,i+1,j-1) :
                  ((i==(nx_d-1) && j>0) ? I2D(nx_d,i-1,j-1) :
                  ((i<(nx_d-1)  && j==0)? I2D(nx_d,i+1,j+1) : I2D(nx_d,i-1,j+1) ) ) ;

        int NWxy = (i>0  && j<(ny_d-1)) ? I2D(nx_d,i-1,j+1) :
                  ((i==0 && j<(ny_d-1)) ? I2D(nx_d,i+1,j+1) :
                  ((i>0  && j==(ny_d-1))? I2D(nx_d,i-1,j-1) : I2D(nx_d,i+1,j-1) ) ) ;

        int NExy = (i<(nx_d-1)  && j<(ny_d-1)) ? I2D(nx_d,i+1,j+1) :
                  ((i==(nx_d-1) && j<(ny_d-1)) ? I2D(nx_d,i-1,j+1) :
                  ((i<(nx_d-1)  && j==(ny_d-1))? I2D(nx_d,i+1,j-1) : I2D(nx_d,i-1,j-1) ) ) ;

        REAL b_S = (j > 0 )? 0.0:
                  ((j==0 && (i==0 || i==(nx_d-1)))? 0.0:
                  rby_d*(g_in.u[I2D(nx_d,i+1,j)] - g_in.u[I2D(nx_d,i-1,j)])) ;

        REAL b_N = (j < (ny_d-1))? 0.0:
                  ((j==(ny_d-1) && (i==0 || i==(nx_d-1)))? 0.0:
                  -rby_d*(g_in.u[I2D(nx_d,i+1,j)] - g_in.u[I2D(nx_d,i-1,j)])) ;

        REAL b_W = (i > 0 )? 0.0:
                  ((i==0 && (j==0 || j==(ny_d-1)))? 0.0:
                  rbx_d*(g_in.u[I2D(nx_d,i,j+1)] - g_in.u[I2D(nx_d,i,j-1)])) ;

        REAL b_E = (i < (nx_d-1))? 0.0:
                  ((i==(nx_d-1) && (j==0 || j==(ny_d-1)))? 0.0:
                  -rbx_d*(g_in.u[I2D(nx_d,i,j+1)] - g_in.u[I2D(nx_d,i,j-1)])) ;

        du2dt += (
                 ( b_S + b_N )*ry_d
             +   ( b_W + b_E )*rx_d  );

        // Correcion to SW SE NW NE boundary conditions
        REAL b_SW = (i>0  && j>0)?  0.0 :
                   ((i==0 && j>1)?  rbx_d*(g_in.u[i2d] - g_in.u[I2D(nx_d,i,j-2)]) :
                   ((i>1  && j==0)? rby_d*(g_in.u[i2d] - g_in.u[I2D(nx_d,i-2,j)]) : 0.0)) ;

        REAL b_SE = (i<(nx_d-1)  && j>0)?  0.0 :
                   ((i==(nx_d-1) && j>1)? -rbx_d*(g_in.u[i2d] - g_in.u[I2D(nx_d,i,j-2)]) :
                   ((i<(nx_d-2)  && j==0)? rby_d*(g_in.u[I2D(nx_d,i+2,j)] - g_in.u[i2d]) : 0.0)) ;

        REAL b_NW = (i>0  && j<(ny_d-1)) ?  0.0 :
                   ((i==0 && j<(ny_d-2)) ?  rbx_d*(g_in.u[I2D(nx_d,i,j+2)] - g_in.u[i2d]) :
                   ((i>1  && j==(ny_d-1))? -rby_d*(g_in.u[i2d] - g_in.u[I2D(nx_d,i-2,j)]) : 0.0)) ;

        REAL b_NE = (i<(nx_d-1)  && j<(ny_d-1)) ? 0.0 :
                   ((i==(nx_d-1) && j<(ny_d-2)) ? -rbx_d*(g_in.u[I2D(nx_d,i,j+2)] - g_in.u[i2d]) :
                   ((i<(nx_d-2)  && j==(ny_d-1))? -rby_d*(g_in.u[I2D(nx_d,i+2,j)] - g_in.u[i2d]) : 0.0)) ;

        du2dt += ( rxy_d * ( (g_in.u[SWxy] + b_SW) +
                             (g_in.u[NExy] + b_NE) -
                             (g_in.u[SExy] + b_SE) -
                             (g_in.u[NWxy] + b_NW) ) );

        if ( gateDiff_d ) {

          // Gate 1
          REAL b_S = (j > 0 )? 0.0:
                    ((j==0 && (i==0 || i==(nx_d-1)))? 0.0:
                    rby_d*(g_in.v[I2D(nx_d,i+1,j)] - g_in.v[I2D(nx_d,i-1,j)])) ;

          REAL b_N = (j < (ny_d-1))? 0.0:
                    ((j==(ny_d-1) && (i==0 || i==(nx_d-1)))? 0.0:
                    -rby_d*(g_in.v[I2D(nx_d,i+1,j)] - g_in.v[I2D(nx_d,i-1,j)])) ;

          REAL b_W = (i > 0 )? 0.0:
                    ((i==0 && (j==0 || j==(ny_d-1)))? 0.0:
                    rbx_d*(g_in.v[I2D(nx_d,i,j+1)] - g_in.v[I2D(nx_d,i,j-1)])) ;

          REAL b_E = (i < (nx_d-1))? 0.0:
                    ((i==(nx_d-1) && (j==0 || j==(ny_d-1)))? 0.0:
                    -rbx_d*(g_in.v[I2D(nx_d,i,j+1)] - g_in.v[I2D(nx_d,i,j-1)])) ;

          dv2dt += (
                   ( b_S + b_N )*ry_d
               +   ( b_W + b_E )*rx_d  );

          // Correcion to SW SE NW NE boundary conditions
          REAL b_SW = (i>0  && j>0)?  0.0 :
                     ((i==0 && j>1)?  rbx_d*(g_in.v[i2d] - g_in.v[I2D(nx_d,i,j-2)]) :
                     ((i>1  && j==0)? rby_d*(g_in.v[i2d] - g_in.v[I2D(nx_d,i-2,j)]) : 0.0)) ;

          REAL b_SE = (i<(nx_d-1)  && j>0)?  0.0 :
                     ((i==(nx_d-1) && j>1)? -rbx_d*(g_in.v[i2d] - g_in.v[I2D(nx_d,i,j-2)]) :
                     ((i<(nx_d-2)  && j==0)? rby_d*(g_in.v[I2D(nx_d,i+2,j)] - g_in.v[i2d]) : 0.0)) ;

          REAL b_NW = (i>0  && j<(ny_d-1)) ?  0.0 :
                     ((i==0 && j<(ny_d-2)) ?  rbx_d*(g_in.v[I2D(nx_d,i,j+2)] - g_in.v[i2d]) :
                     ((i>1  && j==(ny_d-1))? -rby_d*(g_in.v[i2d] - g_in.v[I2D(nx_d,i-2,j)]) : 0.0)) ;

          REAL b_NE = (i<(nx_d-1)  && j<(ny_d-1)) ? 0.0 :
                     ((i==(nx_d-1) && j<(ny_d-2)) ? -rbx_d*(g_in.v[I2D(nx_d,i,j+2)] - g_in.v[i2d]) :
                     ((i<(nx_d-2)  && j==(ny_d-1))? -rby_d*(g_in.v[I2D(nx_d,i+2,j)] - g_in.v[i2d]) : 0.0)) ;

          dv2dt += ( rxy_d * ( (g_in.v[SWxy] + b_SW) +
                               (g_in.v[NExy] + b_NE) -
                               (g_in.v[SExy] + b_SE) -
                               (g_in.v[NWxy] + b_NW) )*rscale_d );

        }

      }

    }

  } else { // Dirichlet bc

    int S = I2D(nx_d,i,j-1);
    int N = I2D(nx_d,i,j+1);
    int W = I2D(nx_d,i-1,j);
    int E = I2D(nx_d,i+1,j);

    if ( circle_d ) {

      bool sc = solid[i2d];
      bool sw = solid[W];
      bool se = solid[E];
      bool sn = solid[N];
      bool ss = solid[S];

      REAL uS = sc && ss ? g_in.u[S] : boundaryVal_d;
      REAL uN = sc && sn ? g_in.u[N] : boundaryVal_d;
      REAL uW = sc && sw ? g_in.u[W] : boundaryVal_d;
      REAL uE = sc && se ? g_in.u[E] : boundaryVal_d;

      du2dt = (
          ( uW - 2.0*u + uE )*rx_d
      +   ( uN - 2.0*u + uS )*ry_d );

      dv2dt = 0.0;

      if ( gateDiff_d ) {

        REAL vS = sc && ss ? g_in.v[S] : boundaryVal_d;
        REAL vN = sc && sn ? g_in.v[N] : boundaryVal_d;
        REAL vW = sc && sw ? g_in.v[W] : boundaryVal_d;
        REAL vE = sc && se ? g_in.v[E] : boundaryVal_d;

        dv2dt = (
             ( vW - 2.0*v + vE )*rx_d*rscale_d
        +    ( vN - 2.0*v + vS )*ry_d*rscale_d );

      }

      if ( anisotropy_d ) {

        int SW = I2D(nx_d,i-1,j-1);
        int SE = I2D(nx_d,i+1,j-1);
        int NW = I2D(nx_d,i-1,j+1);
        int NE = I2D(nx_d,i+1,j+1);

        bool ssw = solid[SW];
        bool sse = solid[SE];
        bool snw = solid[NW];
        bool sne = solid[NE];

        REAL uSWxy = sc && ssw ? g_in.u[SW] : boundaryVal_d ;
        REAL uSExy = sc && sse ? g_in.u[SE] : boundaryVal_d ;
        REAL uNWxy = sc && snw ? g_in.u[NW] : boundaryVal_d ;
        REAL uNExy = sc && sne ? g_in.u[NE] : boundaryVal_d ;

        du2dt += ( rxy_d * (  uSWxy +
                              uNExy -
                              uSExy -
                              uNWxy ) );

        if ( gateDiff_d ) {

          REAL vSWxy = sc && ssw ? g_in.v[SW] : boundaryVal_d ;
          REAL vSExy = sc && sse ? g_in.v[SE] : boundaryVal_d ;
          REAL vNWxy = sc && snw ? g_in.v[NW] : boundaryVal_d ;
          REAL vNExy = sc && sne ? g_in.v[NE] : boundaryVal_d ;

          dv2dt += ( rxy_d * (  vSWxy +
                                vNExy -
                                vSExy -
                                vNWxy )*rscale_d );

        }

      }

    } else { // Square boundary

      REAL uS = j>0 ? g_in.u[S] : boundaryVal_d;
      REAL uN = j<(ny_d-1) ? g_in.u[N] : boundaryVal_d;
      REAL uW = i>0 ? g_in.u[W] : boundaryVal_d;
      REAL uE = i<(nx_d-1) ? g_in.u[E] : boundaryVal_d;

      du2dt = (
           ( uW - 2.0*u + uE )*rx_d
      +    ( uN - 2.0*u + uS )*ry_d );

      dv2dt = 0.0;

      if ( gateDiff_d ) {

        // Gate 1
        REAL vS = j>0 ? g_in.v[S] : boundaryVal_d;
        REAL vN = j<(ny_d-1) ? g_in.v[N] : boundaryVal_d;
        REAL vW = i>0 ? g_in.v[W] : boundaryVal_d;
        REAL vE = i<(nx_d-1) ? g_in.v[E] : boundaryVal_d;

        dv2dt = (
             ( vW - 2.0*v + vE )*rx_d*rscale_d
        +    ( vN - 2.0*v + vS )*ry_d*rscale_d );

      }

      if ( anisotropy_d ) {

        REAL uSWxy = (i>0) && (j>0) ? g_in.u[I2D(nx_d,i-1,j-1)] : boundaryVal_d ;
        REAL uSExy = (i<(nx_d-1)) && (j>0) ? g_in.u[I2D(nx_d,i+1,j-1)] : boundaryVal_d ;
        REAL uNWxy = (i>0) && (j<(ny_d-1)) ? g_in.u[I2D(nx_d,i-1,j+1)] : boundaryVal_d ;
        REAL uNExy = (i<(nx_d-1)) && (j<(ny_d-1)) ? g_in.u[I2D(nx_d,i+1,j+1)] : boundaryVal_d ;

        du2dt += ( rxy_d * (  uSWxy +
                              uNExy -
                              uSExy -
                              uNWxy ) );

        if ( gateDiff_d ) {

          REAL vSWxy = (i>0) && (j>0) ? g_in.v[I2D(nx_d,i-1,j-1)] : boundaryVal_d ;
          REAL vSExy = (i<(nx_d-1)) && (j>0) ? g_in.v[I2D(nx_d,i+1,j-1)] : boundaryVal_d ;
          REAL vNWxy = (i>0) && (j<(ny_d-1)) ? g_in.v[I2D(nx_d,i-1,j+1)] : boundaryVal_d ;
          REAL vNExy = (i<(nx_d-1)) && (j<(ny_d-1)) ? g_in.v[I2D(nx_d,i+1,j+1)] : boundaryVal_d ;

          dv2dt += ( rxy_d * (  vSWxy +
                                vNExy -
                                vSExy -
                                vNWxy )*rscale_d );
        }

      }

    }

  }

  /*------------------------------------------------------------------------
  * RHS
  *------------------------------------------------------------------------
  */

  du2dt -= dt_d*I_sum ;
  dv2dt -= dt_d*I_v ;

  if ( circle_d ) {

    bool sc = solid[i2d];

    /*------------------------------------------------------------------------
    * Euler time integration
    *------------------------------------------------------------------------
    */

    u += tc_d*du2dt ;
    g_out.u[i2d] = sc ? u : 0.0;

    v += tc_d*dv2dt ;
    g_out.v[i2d] = sc ? v : 0.0;

    /*------------------------------------------------------------------------
    * Calculate velocity tangent
    *------------------------------------------------------------------------
    */

    velTan.u[i2d] = sc ? du2dt / dt_d : 0.0;
    velTan.v[i2d] = sc ? dv2dt / dt_d : 0.0;

  } else {

    /*------------------------------------------------------------------------
    * Euler time integration
    *------------------------------------------------------------------------
    */

    u += tc_d*du2dt ;
    g_out.u[i2d] = u;

    v += tc_d*dv2dt ;
    g_out.v[i2d] = v ;

    /*------------------------------------------------------------------------
    * Calculate velocity tangent
    *------------------------------------------------------------------------
    */

    velTan.u[i2d] = du2dt / dt_d;
    velTan.v[i2d] = dv2dt / dt_d;

  }

}

}

void reactionDiffusion_wrapper(size_t pitch, dim3 grid2D, dim3 block2D,
   stateVar gOut_d, stateVar gIn_d,
   stateVar velTan, bool reduceSym, bool *solid) {

  //////////////////////////////////////////////////
  // Notice that every function call we advance 2*dt
  // This can be changed to 1*dt by swapping pointers, but it's slower
  //////////////////////////////////////////////////

  reactionDiffusion_kernel<<<grid2D, block2D>>>(pitch, gOut_d, gIn_d,
    velTan, reduceSym, solid);
  CudaCheckError();

/*
  Karma_kernel<<<grid2D, block2D>>>(pitch, gIn_d, gOut_d,
    solid_d);
  CudaCheckError();
*/
  

}



