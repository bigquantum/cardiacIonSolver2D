
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string.h>
#include <dirent.h>
#include <errno.h>

#include "typeDefinition.cuh"
// #include "globalVariables.cuh"
#include "hostPrototypes.h"

extern paramVar param;

paramVar startMenu(fileVar *strAdress, paramVar param) {


	char strPath[] = "./DATA/"; // Global path
	char strDirR[] = "initCond/sym06r/dataSpiral.dat"; // Read
	char strDirRcsv[] = "initCond/sym01r/dataparamcsv.csv";
	char strDirW[] = "results/sym01/"; // Write

	/*------------------------------------------------------------------------
	* Directory path
	*------------------------------------------------------------------------
	*/

	memcpy(strAdress->read,strPath,sizeof(strAdress->read));
	memcpy(strAdress->readcsv,strPath,sizeof(strAdress->read));
	memcpy(strAdress->p1D,strPath,sizeof(strAdress->p1D));
	memcpy(strAdress->p2D,strPath,sizeof(strAdress->p2D));
	memcpy(strAdress->p2DG,strPath,sizeof(strAdress->p2DG));
	memcpy(strAdress->tip1,strPath,sizeof(strAdress->tip1));
	memcpy(strAdress->tip2,strPath,sizeof(strAdress->tip2));
	memcpy(strAdress->sym,strPath,sizeof(strAdress->sym));
	memcpy(strAdress->contour1,strPath,sizeof(strAdress->contour1));
	memcpy(strAdress->param1,strPath,sizeof(strAdress->param1));
	memcpy(strAdress->param2,strPath,sizeof(strAdress->param2));

	strcat(strAdress->read,strDirR);
	strcat(strAdress->readcsv,strDirRcsv);
	strcat(strAdress->p1D,strDirW);
	strcat(strAdress->p2D,strDirW);
	strcat(strAdress->p2DG,strDirW);
	strcat(strAdress->tip1,strDirW);
	strcat(strAdress->tip2,strDirW);
	strcat(strAdress->sym,strDirW);
	strcat(strAdress->contour1,strDirW);
	strcat(strAdress->param1,strDirW);
	strcat(strAdress->param2,strDirW);

	printf("Do you want to LOAD a previous simulation?\n");
	printf("PRESS 'y' to accept, 'n' to start with a default setup, or any key to EXIT program\n");
	char stryesno;
	scanf(" %c", &stryesno);
	if (stryesno == 'y') {
		printf("Loading parameters\n");
		param.load = true;
		param = loadParamValues(strAdress->readcsv,param);

		// Uncomment to edit the loaded parameters
		
		// param.recordTip = false;
		// param.recordContour = false;
		// param.stimulate = false;
		// param.reduceSym = false;
		// param.reduceSymStart = param.reduceSym;

	} else if (stryesno == 'n') {
		printf("Using default setup\n");
		param.load = false;
		param = parameterSetup(param);
    } else {
        printf("EXIT PROGRAM\n");
        exit(1);
    }

	printf("Do you want to SAVE your data?\n");
	printf("PRESS 'y' to accept, 'n' to decline, or any key to EXIT program\n");
	char stryesno2;
	scanf(" %c", &stryesno2);
	if (stryesno2 == 'y') {
		param.save = true;
	} else if (stryesno2 == 'n') {
		param.save = false;
		pressEnterKey();
    } else {
        printf("EXIT PROGRAM\n");
        exit(1);
    }

	return param;

}

paramVar parameterSetup(paramVar param) {

	/*------------------------------------------------------------------------
	* Default switches
	*------------------------------------------------------------------------
	*/

	param.animate = true;
	param.recordTip = false;
	param.recordContour = false;
	param.stimulate = false;
	param.reduceSym = false;
	param.reduceSymStart = param.reduceSym;

	param.circle = 0;
	param.neumannBC = 1;
	param.gateDiff = 1;
	param.bilinear = 1;
	param.anisotropy = 0;
	param.tipGrad = 0;

	param.clock = 0;
	param.counterclock = 1;
	param.firstFPS = 1;
	param.firstIterContour = 1;

	param.nx = 512;
	param.ny = 512;
	param.Lx = 7.0;
	param.Ly = 7.0;
	param.hx = param.Lx/(param.nx-1.0);
	param.hy = param.Ly/(param.ny-1.0);
	param.dt = 0.01;
	param.dt = param.reduceSym ? param.dt/2.0 : param.dt;
	param.diff_par = 0.001;
	param.diff_per = 0.001;

	param.degrad = 0.0; // Fiber rotation degrees (60)
	REAL theta = param.degrad*pi/180.0;
	param.Dxx = param.diff_par*cos(theta)*cos(theta) +
	param.diff_per*sin(theta)*sin(theta);
	param.Dyy = param.diff_par*sin(theta)*sin(theta) +
	param.diff_per*cos(theta)*cos(theta);
	param.Dxy = (param.diff_par - param.diff_per)*sin(theta)*cos(theta);

	// Anisotropic paramters
	param.rxy = 2.0*param.Dxy*param.dt/(4.0*param.hx*param.hy);
	param.rbx = param.hx*param.Dxy/(param.Dxx*param.hy);
	param.rby = param.hy*param.Dxy/(param.Dyy*param.hx);
	param.rx = param.dt*param.Dxx/(param.hx*param.hx);
	param.ry = param.dt*param.Dyy/(param.hy*param.hy);
	param.rscale = 0.1;
	param.invdx = 0.5/param.hx;
	param.invdy = 0.5/param.hy;

	// Global counter and time
	param.count = 0;
	param.physicalTime = 0.0;
	param.physicalTimeLim = 20000.0;

	// Single cell recordings
	param.eSize = 2; // Number of electrodes
	param.point = make_int2( param.nx/2, param.ny/2 );

	param.stimPeriod = floor(240.0/param.dt); // The number inside is in miliseconds
	param.stimMag = 2.0;

	param.nc = 100; // Numbre of points for circles
	param.stcx = 0.25*param.Lx; // Stimulus position
  	param.stcy = 0.25*param.Ly;
  	param.rdomStim = 0.03*param.Lx; // Stimulus area radius
	param.rdomAPD = 0.15*param.Lx; // APD area radius (always greater than rdomStim)
	param.tipOffsetX = 163; // 163 (512^2) good!
	param.tipOffsetY = 163;
	param.rdomTrapz = 0.5*((param.tipOffsetX+param.tipOffsetY)*param.hx);

	param.boundaryVal = 0.0;

	param.itPerFrame = 200;
	param.sample = (int)1.0/param.dt; // Sample every 1 second

	param.minVarColor = -0.1f;
	param.maxVarColor = 4.0f;
	param.wnx = 512;
	param.wny = 512;
	param.uMax = 4.0f;
	param.uMin = -0.1f;
	param.vMax = 2.0f;
	param.vMin = -0.1f;

	/*========================================================================
	 * Model parameters defined as macros
	 *========================================================================
	*/

	param.Uth = 2.0;
	param.tc = 1.0;
	param.tau_n = 290.0;
	param.tau_e = 3.5;
	param.e_h = 3.0;
	param.e_n = 1.0;
	param.e_star = 1.5415;
	param.Re = 0.35;
	param.M_d = 1.0;
	param.expRe = expf(-param.Re);

	return param;

}


fileVar saveFileNames(fileVar strAdress, paramVar *param) {

	/*------------------------------------------------------------------------
	* Create directory
	*------------------------------------------------------------------------
	*/

	DIR* dir = opendir(strAdress.p1D);
	if (dir) {
	    printf("The saving directory already exists.\n");
	    closedir(dir);
	    printf("Do you want to OVERWRITE the directory %s ?\n",strAdress.p1D);
	    printf("PRESS 'y' to accept or any key to EXIT program\n");
	    char stryes;
	    scanf(" %c", &stryes);
	    if (stryes == 'y') {
	    	int r = remove_directory(strAdress.p1D);
	    	int check;
			check = mkdir( strAdress.p1D , 0700);
	    	if (!check) printf("The directory has been overwritten\n");
	    } else {
	        printf("EXIT PROGRAM\n");
	        exit(1);
	    }
	} else if (ENOENT == errno) {
	    int check;
		check = mkdir( strAdress.p1D , 0700);
		if (!check) printf("NEW directory created: %s\n",strAdress.p1D); 
	} else {
	    /* opendir() failed for some other reason. */
	}

   	/*------------------------------------------------------------------------
	* Select saving package
	*------------------------------------------------------------------------
	*/

	printf("Enter saving Package value:\n");
	printf("--0-No save --1-print2D --2-printTip --3-Symmetry --4-Contour --5-print2D-Multi\n");
   	scanf("%d", &param->savePackage);

   	/*------------------------------------------------------------------------
	* Choose saving options
	*------------------------------------------------------------------------
	*/

	// Add function to not save1!!!!!!!!!!!!!!!!!!!!

	switch (param->savePackage) {

		case 0:
			printf("You have selected Package %d\n", param->savePackage);
			printf("No files will be saved (except the parameter setup)\n");
			printf("\n");

			pressEnterKey();

			param->recordTip = false;
			param->recordContour = false;
			param->stimulate = false;
			param->reduceSym = false;
			param->reduceSymStart = param->reduceSym;

		break;

		case 1:
			printf("You have selected Package %d\n", param->savePackage);
			printf(" -print2D2column\n");
			printf("\n");

			pressEnterKey();

			strcat(strAdress.p2D, "raw_data.dat");

			param->recordTip = false;
			param->recordContour = false;
			param->stimulate = false;
			param->reduceSym = false;
			param->reduceSymStart = param->reduceSym;

		break;

		case 2:
			printf("You have selected Package %d\n", param->savePackage);
			printf(" -printTip\n");
			printf("\n");

			pressEnterKey();

			strcat(strAdress.tip1, "dataTip.dat");
			strcat(strAdress.tip2, "dataTipSize.dat");

			param->recordTip = true;
			param->recordContour = false;
			param->stimulate = false;
			param->reduceSym = false;
			param->reduceSymStart = param->reduceSym;

		break;

		case 3:
			printf("You have selected Package %d\n", param->savePackage);
			printf(" -printTip\n");
			printf(" -printSym\n");
			printf("\n");

			pressEnterKey();

			strcat(strAdress.tip1, "dataTip_sym.dat");
			strcat(strAdress.tip2, "dataTipSize_sym.dat");
			strcat(strAdress.sym, "c_phi_list_sym.dat");

			param->recordTip = true;
			param->recordContour = false;
			param->stimulate = false;
			param->reduceSym = true;
			param->reduceSymStart = param->reduceSym;

		break;

		case 4:
			printf("You have selected Package %d\n", param->savePackage);
			printf(" -printContour\n");
			printf("\n");

			pressEnterKey();

			strcat(strAdress.contour1, "dataContour.dat");
			strcat(strAdress.contour2, "dataContourSize.dat");

			param->recordTip = false;
			param->recordContour = true;
			param->stimulate = true;
			param->reduceSym = false;
			param->reduceSymStart = param->reduceSym;

		break;

		case 5:
			printf("You have selected Package %d\n", param->savePackage);
			printf(" -print2D2column ( Multiple iterations )\n");
			printf("\n");

			pressEnterKey();

			// strcat(strAdress.p2D, "raw_data.dat");
			param->sample = (int)5.0/(param->dt); // Sample every 1 second

			param->recordTip = true;
			param->recordContour = false;
			param->stimulate = false;
			param->reduceSym = true;
			param->reduceSymStart = param->reduceSym;

		break;

		default:
      		printf("No function assigned to this key\n");
    	break;

	}

	return strAdress;

}

void saveFile(fileVar strAdress, paramVar param, stateVar gate_h,
	std::vector<electrodeVar> &electrode, int dt, int *tip_count,
	vec5dyn *tip_vector, std::vector<REAL3> &clist, std::vector<REAL3> &philist,
	bool *firstIter, int *contour_count, float3 *contour_vector) {

   	/*------------------------------------------------------------------------
	* Save data in files
	*------------------------------------------------------------------------
	*/

	switch (param.savePackage) {

		case 0:

		break;

		case 1:
			print2D2column(strAdress.p2D,gate_h);
		break;

		case 2:
			printTip(strAdress.tip1,strAdress.tip2,tip_count,tip_vector);
		break;

		case 3:
			printTip(strAdress.tip1,strAdress.tip2,tip_count,tip_vector);
			printSym(strAdress.sym,clist,philist);
		break;

		case 4:
			printContour(strAdress.contour1,strAdress.contour2,firstIter,
				contour_count,contour_vector);
		break;

		case 5:
			char strCount[32];
			sprintf(strCount, "raw_data_sym%d.dat", param.count);
			strcat(strAdress.p2D, strCount);
			print2D2column(strAdress.p2D,gate_h);
			printf("File %d\n", param.count);

		break;

		default:
      		printf("No function assigned to this key\n");
    	break;

		// print1D(strAdress,gate_h);
		// print2DGnuplot(strAdress,gate_h);
		// print2D2column(gate_h,count,strAdress,sbytes);
		// printVoltageInTime(strAdress,electrode,dt,param.itPerFrame);

	}

}

void loadData(stateVar g_h, fileVar strAdress) {

  /*------------------------------------------------------------------------
  * Load initial conditions
  *------------------------------------------------------------------------
  */

  int i, j, idx;
  float u, v;

  //Print data
  FILE *fp1;
  fp1 = fopen(strAdress.read,"r");

  if (fp1==NULL) {
    puts("Error: can't open the initial condition file\n");
    exit(0);
  }

  for (j=0;j<param.ny;j++) {
    for (i=0;i<param.nx;i++) {
      idx = i + param.nx * j;
      fscanf(fp1, "%f\t%f", &u, &v);
      g_h.u[idx] = u;
      g_h.v[idx] = v;
    }
  }

  fclose(fp1);

}

paramVar loadParamValues(const char *path, paramVar param) {

	FILE *fp = fopen(path, "r");
	if (fp == NULL) {
		perror("Unable to open the initial condition csv file");
		exit(1);
	}

	// Count the numbeer of lines
	int ch = 0;
	int nelements = 0;
	while(!feof(fp))
		{
	  	ch = fgetc(fp);
	  	if(ch == '\n') {
	    	nelements++;
	  	}
	}

	float *values;
	values = (float*)malloc(nelements*sizeof(float));

	// Read elements of csv file
	fp = fopen(path, "r");
	int i, l;
	char line[200];

	l = 0;

	while (fgets(line, sizeof(line), fp)) {
		char *token;
		token = strtok(line, ",");

		// Read rows
		for (i=0;i<2;i++) {
			// Read second element of each row
			if (i == 1) {
				values[l] = strtof(token, NULL);
				// printf("%s",token);
			}
			if ( (token != NULL) ) {
				token = strtok(NULL,",");
			}
		}

		l++;
	}

	printf("Elements = %d\n",nelements);

	/*------------------------------------------------------------------------
	* Load parameters
	*------------------------------------------------------------------------
	*/

	param.animate = true;
	param.recordTip = (int)values[2];
	param.recordContour = (int)values[3];
	param.stimulate = (int)values[4];
	param.reduceSym = (int)values[5];
	param.reduceSymStart = (int)param.reduceSym;

	param.circle = (int)values[6];
	param.neumannBC = (int)values[7];
	param.gateDiff = (int)values[8];
	param.bilinear = (int)values[9];
	param.anisotropy = (int)values[10];
	param.tipGrad = (int)values[11];

	param.clock = (int)values[12];
	param.counterclock = (int)values[13];

	param.nx = (int)values[14];
	param.ny = (int)values[15];
	param.Lx = values[16];
	param.Ly = values[17];
	param.hx = values[18];
	param.hy = values[19];
	param.dt = param.reduceSym ? values[20]/2.0f : values[20];
	param.diff_par = values[21];
	param.diff_per = values[22];

	param.degrad = values[23]; // Fiber rotation degrees (60)
	param.Dxx = values[24];
	param.Dyy = values[25];
	param.Dxy = values[26];

	// Anisotropic paramters
	param.rxy = values[27];
	param.rbx = values[28];
	param.rby = values[29];
	param.rx = values[30];
	param.ry = values[31];
	param.rscale = values[32];
	param.invdx = values[33];
	param.invdy = values[34];

	// Global counter and time
	param.count = 0;
	param.physicalTime = 0.0; // Start timer /// FIXME !!!!!!!!!!!!!!!!!!!!!!!!!! numbers
	param.physicalTimeLim = values[35];

	// Single cell recordings
	param.eSize = (int)values[36]; // Number of electrodes
	param.point = make_int2( (int)values[37], (int)values[38] );

	param.stimPeriod = (int)values[39]; // The number inside is in miliseconds
	param.stimMag = values[40];

	param.nc = (int)values[41]; // Numbre of points for circles
	param.stcx = values[42]; // Stimulus position
  	param.stcy = values[43];
  	param.rdomStim = values[44]; // Stimulus area radius
	param.rdomAPD = values[45]; // APD area radius (always greater than rdomStim)
	param.tipOffsetX = values[46];
	param.tipOffsetY = values[47];
	param.rdomTrapz = values[48];

	param.boundaryVal = values[49];

	param.itPerFrame = values[50];
	param.sample = (int)values[51]; // Sample every 1 second

	param.minVarColor = values[52];
	param.maxVarColor = values[53];
	param.wnx = (int)values[54];
	param.wny = (int)values[55];
	param.uMax = values[56];
	param.uMin = values[57];
	param.vMax = values[58];
	param.vMin = values[59];

	param.Uth = values[60];
	param.tc = values[61];
	param.tau_e = values[62];
	param.tau_n = values[63];
	param.e_h = values[64];
	param.e_n = values[65];
	param.e_star = values[66];
	param.Re = values[67];
	param.M_d = values[68];
	param.expRe = values[69];

	free(values);

	return param;
}

void pressEnterKey(void) {
	// Ask for ENTER key
	printf("Press [Enter] key to continue\n");
	printf("[Ctrl]+[C] to terminate program.\n");
	while(getchar()!='\n'); // option TWO to clean stdin
	getchar(); // wait for ENTER
}

// Taken from: https://stackoverflow.com/questions/2256945/removing-a-non-empty-directory-programmatically-in-c-or-c/2256974
int remove_directory(const char *path) {
   DIR *d = opendir(path);
   size_t path_len = strlen(path);
   int r = -1;

   if (d) {
      struct dirent *p;

      r = 0;
      while (!r && (p=readdir(d))) {
          int r2 = -1;
          char *buf;
          size_t len;

          /* Skip the names "." and ".." as we don't want to recurse on them. */
          if (!strcmp(p->d_name, ".") || !strcmp(p->d_name, ".."))
             continue;

          len = path_len + strlen(p->d_name) + 2; 
          buf = (char*)malloc(len);

          if (buf) {
             struct stat statbuf;

             snprintf(buf, len, "%s/%s", path, p->d_name);
             if (!stat(buf, &statbuf)) {
                if (S_ISDIR(statbuf.st_mode))
                   r2 = remove_directory(buf);
                else
                   r2 = unlink(buf);
             }
             free(buf);
          }
          r = r2;
      }
      closedir(d);
   }

   if (!r)
      r = rmdir(path);

   return r;
}