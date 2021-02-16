#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "typeDefinition.cuh"
//#include "globalVariables.cuh"

#include "./common/CudaSafeCall.h"

extern __device__ vec5dyn tip_vector[TIPVECSIZE];
extern __device__ int tip_count;
extern paramVar param;

// Print a 1D slice of data
void print1D(const char *path, stateVar g_h) {

	int i, j, idx;

	i = param.nx/2;

	//Print data
	FILE *fp1;
	fp1 = fopen(path,"w+");


	// Notice we are not saving the ghost points
	for (j=0;j<param.ny;j++) {
		idx = i + param.nx * j;
		fprintf(fp1, "%d\t %f\n", j, (float)g_h.u[idx]);
	}

	fclose (fp1);

	printf("1D data file created\n");

}

// Print a 2D slice of data
void print2DGnuplot(const char *path, stateVar g_h) {

	int i, j, idx;

	//Print data
	FILE *fp1;
	fp1 = fopen(path,"w+");

	// Notice we are not saving the ghost points
	for (j=0;j<param.ny;j++) {
		for (i=0;i<param.nx;i++) {
			idx = i + param.nx * j;
			fprintf(fp1, "%d\t %d\t %f\n", i, j, (float)g_h.u[idx]);
			}
		fprintf(fp1,"\n");
	}

	fclose (fp1);

	printf("2D GNU format data file created\n");

}

// Print a 2D slice of data
void print2D2column(const char *path, stateVar g_h) {

  int i, j, idx;

  //Print data
  FILE *fp1;
  fp1 = fopen(path,"w+");

  // Notice we are not saving the ghost points
  for (j=0;j<param.ny;j++) {
    for (i=0;i<param.nx;i++) {
      idx = i + param.nx * j;
      fprintf(fp1,"%f %f\n", (float)g_h.u[idx], (float)g_h.v[idx]);
    }
  }

  fclose (fp1);

  printf("2D data file created\n");

}

// Voltage time tracing
void printVoltageInTime(const char *path, std::vector<electrodeVar> &sol, 
  REAL dt, int itPerFrame) {

  int i;

  //Print data
  FILE *fp1;
  fp1 = fopen(path,"w+");

  for (i=0;i<sol.size();i++) {
    fprintf(fp1, "%f\t", i*(float)dt*itPerFrame);
    fprintf(fp1, "%f\t", (float)sol[i].e0);
    fprintf(fp1, "%f\n", (float)sol[i].e1);
    }

  fclose (fp1);

  printf("Voltage time series file created\n");

}

void printTip(const char *path1, const char *path2, int *tip_count, vec5dyn *tip_vector) {

  //Print data
  FILE *fp1, *fp2;

  fp1 = fopen(path1,"w+");
  fp2 = fopen(path2,"w+");

  int *tip_pts;
  tip_pts = (int*)malloc(sizeof(int));
  CudaSafeCall(cudaMemcpy(tip_pts,tip_count,sizeof(int),cudaMemcpyDeviceToHost));

  if (*tip_pts > TIPVECSIZE ) {
    printf("ERROR: NUMBER OF TIP POINTS EXCEEDS tip_vector SIZE\n");
    exit(0);
  }

  vec5dyn *tip_array;
  tip_array = (vec5dyn*)malloc((*tip_pts)*sizeof(vec5dyn));
  CudaSafeCall(cudaMemcpy(tip_array,tip_vector,(*tip_pts)*sizeof(vec5dyn),cudaMemcpyDeviceToHost));

  if (*tip_pts > 0) {
    for (size_t i = 0;i<(*tip_pts);i++) {
      fprintf(fp1,"%f %f %f %f %f\n",tip_array[i].x,tip_array[i].y,
        tip_array[i].vx,tip_array[i].vy,tip_array[i].t);
    }
    fprintf(fp2,"%d\n", *tip_pts);
  }

  fclose (fp1);
  fclose (fp2);

  free(tip_pts);
  free(tip_array);

  printf("Tip files created\n");

}

void printSym(const char *path, std::vector<REAL3> &clist, std::vector<REAL3> &philist) {

  //Print data
  FILE *fp1;
  fp1 = fopen(path,"w+");

  for (size_t i=0;i<(clist.size());i++) {
    fprintf(fp1,"%f %f %f %f %f %f\n", 
    	clist[i].x,clist[i].y,clist[i].t,philist[i].x,philist[i].y,philist[i].t);
    }

  fclose (fp1);

  printf("Symmetry files created\n");

}

void printContour(const char *path1, const char *path2, bool *firstIter,
  int *contour_count, float3 *contour_vector) {

  //Print data
  FILE *fp1, *fp2;

  if (*firstIter==true) {
    fp1 = fopen(path1,"w+");
    fp2 = fopen(path2,"w+");
    *firstIter = false;
  } else {
    fp1 = fopen(path1,"a+"); // Write on the same data file
    fp2 = fopen(path2,"a+");
  }

  int *contour_pts;
  contour_pts = (int*)malloc(sizeof(int));
  CudaSafeCall(cudaMemcpy(contour_pts,contour_count,sizeof(int),cudaMemcpyDeviceToHost));
  // printf("No. pts = %d\n",*contour_pts);

  if (*contour_pts > param.nx*param.ny ) {
    printf("ERROR: NUMBER OF CONTOUR POINTS EXCEEDS contour_vector SIZE\n");
    exit(0);
  }

  float3 *contour_array;
  contour_array = (float3*)malloc((*contour_pts)*sizeof(float3));
  CudaSafeCall(cudaMemcpy(contour_array,contour_vector,(*contour_pts)*sizeof(float3),cudaMemcpyDeviceToHost));

  if (*contour_pts > 0) {
    for (size_t i = 0;i<(*contour_pts);i++) {
      fprintf(fp1,"%f %f %f\n",contour_array[i].x,contour_array[i].y,contour_array[i].z);
    }
    fprintf(fp2,"%d\n", *contour_pts);
  }

  fclose (fp1);
  fclose (fp2);

  free(contour_pts);
  free(contour_array);

  printf("Contour files created\n");

}

void printParameters(fileVar strAdress, paramVar param) {

  /*------------------------------------------------------------------------
  * Create dat file
  *------------------------------------------------------------------------
  */

  char resultsPath[100];
  char strDirParam[] = "dataparam.dat";
  char strDirParamCSV[] = "dataparamcsv.csv";

  memcpy(resultsPath,strAdress.param1,sizeof(resultsPath));
  strcat(strAdress.param1,strDirParam);
  strcat(strAdress.param2,strDirParamCSV);

  //Print data
  FILE *fp1, *fp2;
  fp1 = fopen(strAdress.param1,"w+");
  fp2 = fopen(strAdress.param2,"w+");

  if ( param.load ) {
    fprintf(fp1,"Initial condition path: %s\n", strAdress.read);
  } else {
    fprintf(fp1,"Initial condition path: %s\n", "NA");
  }

  if ( param.save ) {
    fprintf(fp1,"Results file path: %s\n", resultsPath);
  } else {
    fprintf(fp1,"Results file path: %s\n", "NA");
  }

  fprintf(fp1,"\n********Switches*********\n");
  fprintf(fp1,"Tip recording: %s \n", param.recordTip ? "On" : "Off");
  fprintf(fp1,"Contour recording: %s \n", param.recordContour ? "On" : "Off");
  fprintf(fp1,"Pacing stimulus: %s \n", param.stimulate ? "On" : "Off");
  fprintf(fp1,"Reduce symmetry: %s \n", param.reduceSym ? "On" : "Off");

  fprintf(fp1,"Circular boundary: %s \n", param.circle ? "On" : "Off");
  fprintf(fp1,"Neumann BCs: %s \n", param.neumannBC ? "On" : "Off");
  fprintf(fp1,"Gate diffusion: %s \n", param.gateDiff ? "On" : "Off");
  fprintf(fp1,"Tip bilinear interpolation: %s \n", param.bilinear ? "On" : "Off");
  fprintf(fp1,"Anisotropic tisue: %s \n", param.anisotropy ? "On" : "Off");
  fprintf(fp1,"Tip gradient: %s \n", param.tipGrad ? "On" : "Off");
  fprintf(fp1,"Conduction block clock: %s \n", param.clock ? "On" : "Off");
  fprintf(fp1,"Conduction block counterclock: %s \n", param.counterclock ? "On" : "Off");

  fprintf(fp1,"\n********Grid dimensions*********\n");
  fprintf(fp1,"# grid points X = %d \n", param.nx);
  fprintf(fp1,"# grid points Y = %d \n", param.ny);

  fprintf(fp1,"\n********Spatial dimensions*********\n");
  fprintf(fp1,"Physical Lx length %f cm \n", (float)param.Lx);
  fprintf(fp1,"Physical Ly length %f cm \n", (float)param.Ly);
  fprintf(fp1,"Physical dx %f cm \n", (float)param.hx);
  fprintf(fp1,"Physical dy %f cm \n", (float)param.hy);

  fprintf(fp1,"\n********Time*********\n");
  fprintf(fp1,"Time step; %f ms \n", param.reduceSym ? 2.0*(float)param.dt : param.dt);

  fprintf(fp1,"\n********Diffusion*********\n");
  fprintf(fp1,"Diffusion parallel component: %f cm^2/ms\n", (float)param.diff_par);
  fprintf(fp1,"Diffusion perpendicular component: %f cm^2/ms\n", (float)param.diff_per);

  fprintf(fp1,"\n******Anisotropic tissue*******\n");
  fprintf(fp1,"Initial fiber angle: %f deg\n", (float)param.degrad);
  fprintf(fp1,"Diffusion Dxx: %f cm^2/ms\n", (float)param.Dxx);
  fprintf(fp1,"Diffusion Dyy: %f cm^2/ms\n", (float)param.Dyy);
  fprintf(fp1,"Diffusion Dxy: %f cm^2/ms\n", (float)param.Dxy);
  fprintf(fp1,"rxy (2*Dxy*dt/(4*dx*dy)): %f \n", (float)param.rxy);
  fprintf(fp1,"rbx (hx*Dxy/(Dxx*dy)): %f \n", (float)param.rbx);
  fprintf(fp1,"rby (hy*Dxy/(Dyy*dx)): %f \n", (float)param.rby);

  fprintf(fp1,"rx (Dxx*dt/(dx*dx)): %f \n", (float)param.rx);
  fprintf(fp1,"ry (Dyy*dt/(dy*dy)): %f \n", (float)param.ry);
  fprintf(fp1,"Gate r-scale %f \n", (float)param.rscale);
  fprintf(fp1,"invdx (1/(2*hx)) %f \n", (float)param.invdx);
  fprintf(fp1,"invdy (1/(2*hy)) %f \n", (float)param.invdy);

  fprintf(fp1,"\n******Counting time*******\n");
  fprintf(fp1,"Physical time limit: %f ms \n", (float)param.physicalTimeLim);

  fprintf(fp1,"\n*****Single cell******\n");
  fprintf(fp1,"Number of electrodes %d \n", param.eSize);
  fprintf(fp1,"Electrode position x: %f px\n", (float)param.point.x);
  fprintf(fp1,"Electrode position y: %f px\n", (float)param.point.y);

  fprintf(fp1,"Stimulus period: %d ms \n", param.stimPeriod);
  fprintf(fp1,"Stimulus magnitude: %f \n", (float)param.stimMag);

  fprintf(fp1,"\n*****Areas and figures******\n");
  fprintf(fp1,"Number of points in circles: %d \n",param.nc);
  fprintf(fp1,"Stimulus position x: %f \n",(float)param.stcx);
  fprintf(fp1,"Stimulus position y: %f \n",(float)param.stcy);
  fprintf(fp1,"Stimulus area radius: %f \n",(float)param.rdomStim);
  fprintf(fp1,"APD area radius: %f \n",(float)param.rdomAPD);
  fprintf(fp1,"Tip offset x %d ps \n",param.tipOffsetX);
  fprintf(fp1,"Tip offset y %d ps \n",param.tipOffsetY);
  fprintf(fp1,"Integral area radius: %f \n",(float)param.rdomTrapz);

  fprintf(fp1,"\n********Boundary*********\n");
  fprintf(fp1,"Dirichlet BC value: %f \n",(float)param.boundaryVal);

  fprintf(fp1,"\n********Graphics*********\n");
  fprintf(fp1,"Iterations per frame %d \n", param.itPerFrame);
  fprintf(fp1,"Sampling frequency %d \n", param.sample);

  fprintf(fp1,"Min signal range %f \n", (float)param.minVarColor);
  fprintf(fp1,"Max signal range %f \n", (float)param.maxVarColor);
  fprintf(fp1,"Secondary window size x %d px \n", param.wnx);
  fprintf(fp1,"Secondary window size y %d px \n", param.wny);
  fprintf(fp1,"Secondary window max signal 1 range %f \n", (float)param.uMax);
  fprintf(fp1,"Secondary window min signal 1 range %f \n", (float)param.uMin);
  fprintf(fp1,"Secondary window max signal 2 range %f \n", (float)param.vMax);
  fprintf(fp1,"Secondary window min signal 2 range %f \n", (float)param.vMin);

  fprintf(fp1,"\n********Model parameters*********\n");
  fprintf(fp1,"Filament voltage threshold: %f Volts\n", (float)param.Uth);
  fprintf(fp1,"time scale (tc):  %f\n", (float)param.tc);
  fprintf(fp1,"tau_e:    %f\n", (float)param.tau_e);
  fprintf(fp1,"tau_n:   %f\n", (float)param.tau_n);
  fprintf(fp1,"e_h:  %f\n", (float)param.e_h);
  fprintf(fp1,"e_n:   %f\n", (float)param.e_n);
  fprintf(fp1,"e_star:  %f\n", (float)param.e_star);
  fprintf(fp1,"Re: %f\n", (float)param.Re);
  fprintf(fp1,"M_d: %f\n", (float)param.M_d);
  fprintf(fp1,"expRe:  %f\n", (float)param.expRe);

  fclose (fp1);


  /*------------------------------------------------------------------------
  * Create CSV file
  *------------------------------------------------------------------------
  */


  if ( param.load ) {
    fprintf(fp2,"Initial condition path:,%s\n", strAdress.read);
  } else {
    fprintf(fp2,"Initial condition path:,%s\n", "NA");
  }

  if ( param.save ) {
    fprintf(fp2,"Results file path:,%s\n", resultsPath);
  } else {
    fprintf(fp2,"Results file path:,%s\n", "NA");
  }

  fprintf(fp2,"Tip recording:,%d\n", param.recordTip ? 1 : 0);
  fprintf(fp2,"Contour recording:,%d\n", param.recordContour ? 1 : 0);
  fprintf(fp2,"Pacing stimulus:,%d\n", param.stimulate ? 1 : 0);
  fprintf(fp2,"Reduce symmetry:,%d\n", param.reduceSym ? 1 : 0);

  fprintf(fp2,"Circular boundary:,%d\n", param.circle ? 1 : 0);
  fprintf(fp2,"Neumann BCs:,%d\n", param.neumannBC ? 1 : 0);
  fprintf(fp2,"Gate diffusion:,%d\n", param.gateDiff ? 1 : 0);
  fprintf(fp2,"Tip bilinear interpolation:,%d\n", param.bilinear ? 1 : 0);
  fprintf(fp2,"Anisotropic tisue:,%d\n", param.anisotropy ? 1 : 0);
  fprintf(fp2,"Tip gradient:,%d\n", param.tipGrad ? 1 : 0);
  fprintf(fp2,"Conduction block clock:,%d\n", param.clock ? 1 : 0);
  fprintf(fp2,"Conduction block counterclock:,%d\n", param.counterclock ? 1 : 0);

  fprintf(fp2,"# grid points X =,%d\n", param.nx);
  fprintf(fp2,"# grid points Y =,%d\n", param.ny);

  fprintf(fp2,"Physical Lx length,%f\n", (float)param.Lx);
  fprintf(fp2,"Physical Ly length,%f\n", (float)param.Ly);
  fprintf(fp2,"Physical dx,%f\n", (float)param.hx);
  fprintf(fp2,"Physical dy,%f\n", (float)param.hy);

  fprintf(fp2,"Time step:,%f\n", param.reduceSym ? 2.0*(float)param.dt : param.dt);

  fprintf(fp2,"Diffusion parallel component:,%f\n", (float)param.diff_par);
  fprintf(fp2,"Diffusion perpendicular component:,%f\n", (float)param.diff_per);

  fprintf(fp2,"Initial fiber angle:,%f\n", (float)param.degrad);
  fprintf(fp2,"Diffusion Dxx:,%f\n", (float)param.Dxx);
  fprintf(fp2,"Diffusion Dyy:,%f\n", (float)param.Dyy);
  fprintf(fp2,"Diffusion Dxy:,%f\n", (float)param.Dxy);
  fprintf(fp2,"rxy (2*Dxy*dt/(4*dx*dy)):,%f\n", (float)param.rxy);
  fprintf(fp2,"rbx (hx*Dxy/(Dxx*dy)):,%f\n", (float)param.rbx);
  fprintf(fp2,"rby (hy*Dxy/(Dyy*dx)):,%f\n", (float)param.rby);

  fprintf(fp2,"rx (Dxx*dt/(dx*dx)):,%f\n", (float)param.rx);
  fprintf(fp2,"ry (Dyy*dt/(dy*dy)):,%f\n", (float)param.ry);
  fprintf(fp2,"Gate r-scale:,%f\n", (float)param.rscale);
  fprintf(fp2,"invdx (1/(2*hx)),%f\n", (float)param.invdx);
  fprintf(fp2,"invdy (1/(2*hy)),%f\n", (float)param.invdy);

  fprintf(fp2,"Physical time limit:,%f\n", (float)param.physicalTimeLim);

  fprintf(fp2,"Number of electrodes,%d\n", param.eSize);
  fprintf(fp2,"Electrode position x:,%f\n", (float)param.point.x);
  fprintf(fp2,"Electrode position y:,%f\n", (float)param.point.y);

  fprintf(fp2,"Stimulus period:,%d\n", param.stimPeriod);
  fprintf(fp2,"Stimulus magnitude:,%f\n", (float)param.stimMag);

  fprintf(fp2,"Number of points in circles:,%d\n",param.nc);
  fprintf(fp2,"Stimulus position x:,%f\n",(float)param.stcx);
  fprintf(fp2,"Stimulus position y:,%f\n",(float)param.stcy);
  fprintf(fp2,"Stimulus area radius:,%f\n",(float)param.rdomStim);
  fprintf(fp2,"APD area radius:,%f\n",(float)param.rdomAPD);
  fprintf(fp2,"Tip offset x,%d\n",param.tipOffsetX);
  fprintf(fp2,"Tip offset y,%d\n",param.tipOffsetY);
  fprintf(fp2,"Integral area radius:,%f\n",(float)param.rdomTrapz);

  fprintf(fp2,"Dirichlet BC value:,%f\n",(float)param.boundaryVal);

  fprintf(fp2,"Iterations per frame:,%d\n", param.itPerFrame);
  fprintf(fp2,"Sampling frequency:,%d\n", param.sample);

  fprintf(fp2,"Min signal range,%f\n", (float)param.minVarColor);
  fprintf(fp2,"Max signal range,%f\n", (float)param.maxVarColor);
  fprintf(fp2,"Secondary window size x,%d\n", param.wnx);
  fprintf(fp2,"Secondary window size y,%d\n", param.wny);
  fprintf(fp2,"Secondary window min signal 1 range,%f\n", (float)param.uMin);
  fprintf(fp2,"Secondary window max signal 1 range,%f\n", (float)param.uMax);
  fprintf(fp2,"Secondary window min signal 2 range,%f\n", (float)param.vMax);
  fprintf(fp2,"Secondary window max signal 2 range,%f\n", (float)param.vMin);

  fprintf(fp2,"Filament voltage threshold:,%f\n", (float)param.Uth);
  fprintf(fp2,"time scale (tc):,%f\n", (float)param.tc);
  fprintf(fp2,"tau_e:,%f\n", (float)param.tau_e);
  fprintf(fp2,"tau_n:,%f\n", (float)param.tau_n);
  fprintf(fp2,"e_h:,%f\n", (float)param.e_h);
  fprintf(fp2,"e_n:,%f\n", (float)param.e_n);
  fprintf(fp2,"e_star:,%f\n", (float)param.e_star);
  fprintf(fp2,"Re:,%f\n", (float)param.Re);
  fprintf(fp2,"M_d:,%f\n", (float)param.M_d);
  fprintf(fp2,"expRe:,%f\n", (float)param.expRe);

  fclose (fp2);

  printf("Parameter files created\n");

}