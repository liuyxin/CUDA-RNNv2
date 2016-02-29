#ifndef TRAINNETWORK_H
#define TRAINNETWORK_H
#include "Config.h"
#include "cuMatrix.h"
#include "cuMatrixVector.h"
#include "Samples.h"
#include "InputInit.h"
#include "costGradient.h"
#include "resultPredict.h"
#include <stdio.h>
#include <iostream>
#include <time.h>
#include <mpi.h>
using namespace std;
extern int totalNode;
extern int thisNode;
extern int namelen;
extern char nodeName[];
const int BUFFER_LEN = 600000;
void trainNetwork(vector<HiddenLayer> &Hiddenlayers, SoftMax &SMR,
		int reword_size);
void reduceMat(cuMatrix& cumat,hostPtr& hptr);
	
#endif
