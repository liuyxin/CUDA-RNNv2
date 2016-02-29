#include "TrainNetwork.h"

int totalNode;
int thisNode;
int namelen;
char nodeName[MPI_MAX_PROCESSOR_NAME];

void reduceMat(cuMatrix& cumat,hostPtr& hptr){
	cumat.toCpu();
	int alreadySent = 0;
	float *const src = cumat.getHost();
	float *const dst = hptr.getPtr();
	MPI_Barrier(MPI_COMM_WORLD);
	while(alreadySent < cumat.sizes()){
		int count;
		if(BUFFER_LEN < (cumat.sizes() - alreadySent)){
			count = BUFFER_LEN;
		} else {
			count = cumat.sizes() - alreadySent;
		}
		if( alreadySent & (sizeof(float) - 1)){
			printf("reduceMat :offset error \n");
			exit(0);
		}
		int offset = alreadySent / sizeof(float);
		MPI_Reduce(src+offset, dst+offset, count, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);		
		MPI_Bcast(dst+offset, count, MPI_FLOAT, 0, MPI_COMM_WORLD);
		alreadySent += count;
	}
	cumat.toGpu(hptr);
	cumat /= (float)totalNode;
}

void trainNetwork(vector<HiddenLayer> &Hiddenlayers, SoftMax &SMR,
		int reword_size) {
	cuMatrix v_smr_W_l(SMR.W_l.rows(), SMR.W_l.cols());
	cuMatrix smrW_ld2(SMR.W_l.rows(), SMR.W_l.cols());
	cuMatrix v_smr_W_r(SMR.W_l.rows(), SMR.W_l.cols());
	cuMatrix smrW_rd2(SMR.W_l.rows(), SMR.W_l.cols());
	cuMatrix smr_lr_W(SMR.W_l.rows(), SMR.W_l.cols());

	vector<cuMatrix> v_hl_W_l;
	vector<cuMatrix> hlW_ld2;
	vector<cuMatrix> v_hl_U_l;
	vector<cuMatrix> hlU_ld2;
	vector<cuMatrix> v_hl_W_r;
	vector<cuMatrix> hlW_rd2;
	vector<cuMatrix> v_hl_U_r;
	vector<cuMatrix> hlU_rd2;
	vector<cuMatrix> hidden_lr_W;
	vector<cuMatrix> hidden_lr_U;
	for (int i = 0; i < Hiddenlayers.size(); i++) {
		v_hl_W_l.push_back(
				cuMatrix(Hiddenlayers[i].W_l.rows(),
					Hiddenlayers[i].W_l.cols()));

		v_hl_U_l.push_back(
				cuMatrix(Hiddenlayers[i].U_l.rows(),
					Hiddenlayers[i].U_l.cols()));

		hlW_ld2.push_back(
				cuMatrix(Hiddenlayers[i].W_l.rows(),
					Hiddenlayers[i].W_l.cols()));

		hlU_ld2.push_back(
				cuMatrix(Hiddenlayers[i].U_l.rows(),
					Hiddenlayers[i].U_l.cols()));

		v_hl_W_r.push_back(
				cuMatrix(Hiddenlayers[i].W_l.rows(),
					Hiddenlayers[i].W_l.cols()));

		v_hl_U_r.push_back(
				cuMatrix(Hiddenlayers[i].U_l.rows(),
					Hiddenlayers[i].U_l.cols()));

		hlW_rd2.push_back(
				cuMatrix(Hiddenlayers[i].W_l.rows(),
					Hiddenlayers[i].W_l.cols()));

		hlU_rd2.push_back(
				cuMatrix(Hiddenlayers[i].U_l.rows(),
					Hiddenlayers[i].U_l.cols()));
		hidden_lr_W.push_back(
				cuMatrix(Hiddenlayers[i].W_l.rows(),
					Hiddenlayers[i].W_l.cols()));
		hidden_lr_U.push_back(
				cuMatrix(Hiddenlayers[i].U_l.rows(),
					Hiddenlayers[i].U_l.cols()));
	}
	float Momentum_w = 0.5;
	float Momentum_u = 0.5;
	float Momentum_d2 = 0.5;
	float mu = 1e-2;
	unsigned int maxSize = 0;
	int k = 0;
	bool updateBNL = false;

	for (int i = 0; i < Hiddenlayers.size(); i++) {
		if(maxSize < Hiddenlayers[i].W_l.sizes())
			maxSize = Hiddenlayers[i].W_l.sizes();
		if(maxSize < Hiddenlayers[i].U_l.sizes())
			maxSize = Hiddenlayers[i].U_l.sizes();
	}	
	if(maxSize < SMR.W_l.sizes())
		maxSize = SMR.W_l.sizes();
	hostPtr hptr(maxSize);
	cuMatrix4d acti_0(reword_size,Config::instance()->get_batch_size(),1,Config::instance()->get_ngram());
	cuMatrix sampleY(Config::instance()->get_ngram(),
			Config::instance()->get_batch_size());

	costParamentInit(Hiddenlayers, SMR);

	printf("process %d on %s ,start training network.\n",thisNode,nodeName);
	time_t trainBegin, trainEnd;
	trainBegin = time(NULL);
	for (int epo = 1; epo <= Config::instance()->get_training_epochs(); epo++) {
		for (; k <= Config::instance()->get_iter_per_epo() * epo; k++) {
			if (0 == (k+1) % 200){
				updateBNL = true;
			}
			else{
				updateBNL = false;
			}
			if (k > 30) {
				Momentum_w = 0.95;
				Momentum_u = 0.95;
				Momentum_d2 = 0.90;
			}
			if(0 == thisNode){
				cout << "epoch: " << epo << ", iter: " << k << endl;
			}
			init_acti0(acti_0, sampleY);
			getNetworkCost(acti_0, sampleY, Hiddenlayers, SMR, updateBNL);
			smrW_ld2.Mul2(Momentum_d2, smrW_ld2);

			smrW_ld2 += SMR.W_ld2 * (1.0 - Momentum_d2);
			cuDiv(SMR.lr_W, (smrW_ld2 + mu), smr_lr_W);

			v_smr_W_l.Mul2(Momentum_w, v_smr_W_l);
			v_smr_W_l += SMR.W_lgrad.Mul(smr_lr_W) * (1.0 - Momentum_w);
			SMR.W_l -= v_smr_W_l;
			if(0 == ((k+1) % 300)){
				reduceMat(SMR.W_l, hptr);
			}
			smrW_rd2.Mul2(Momentum_d2, smrW_rd2);
			smrW_rd2 += SMR.W_rd2 * (1.0 - Momentum_d2);
			cuDiv(SMR.lr_W, smrW_rd2 + mu, smr_lr_W);

			v_smr_W_r.Mul2(Momentum_w, v_smr_W_r);
			v_smr_W_r += SMR.W_rgrad.Mul(smr_lr_W) * (1.0 - Momentum_w);
			SMR.W_r -= v_smr_W_r;
			if(0 == ((k+1) % 300)){
				reduceMat(SMR.W_r,hptr);
			}
			// hidden layer update

			for (int i = 0; i < Hiddenlayers.size(); i++) {
				hlW_ld2[i] *= Momentum_d2;
				hlW_ld2[i] += Hiddenlayers[i].W_ld2 * (1.0 - Momentum_d2);
				cuDiv(Hiddenlayers[i].lr_W, hlW_ld2[i] + mu, hidden_lr_W[i]);
				v_hl_W_l[i] *= Momentum_w;
				v_hl_W_l[i] += Hiddenlayers[i].W_lgrad.Mul(hidden_lr_W[i])
					* (1.0 - Momentum_w);
				Hiddenlayers[i].W_l -= v_hl_W_l[i];

				if(0 ==((k+1) % 300)){
					reduceMat(Hiddenlayers[i].W_l, hptr);
				}

				hlU_ld2[i] *= Momentum_d2;
				hlU_ld2[i] += Hiddenlayers[i].U_ld2 * (1.0 - Momentum_d2);
				cuDiv(Hiddenlayers[i].lr_U, hlU_ld2[i] + mu, hidden_lr_U[i]);
				v_hl_U_l[i] *= Momentum_u;
				v_hl_U_l[i] += Hiddenlayers[i].U_lgrad.Mul(hidden_lr_U[i])
					* (1.0 - Momentum_u);
				Hiddenlayers[i].U_l -= v_hl_U_l[i];
				if(0 ==((k+1) % 300)){
					reduceMat(Hiddenlayers[i].U_l, hptr);
				}

				hlW_rd2[i] *= Momentum_d2;
				hlW_rd2[i] += Hiddenlayers[i].W_rd2 * (1.0 - Momentum_d2);
				cuDiv(Hiddenlayers[i].lr_W, hlW_rd2[i] + mu, hidden_lr_W[i]);
				v_hl_W_r[i] *= Momentum_w;
				v_hl_W_r[i] += Hiddenlayers[i].W_rgrad.Mul(hidden_lr_W[i])
					* (1.0 - Momentum_w);
				Hiddenlayers[i].W_r -= v_hl_W_r[i];
				if(0 ==((k+1) % 300)){
					reduceMat(Hiddenlayers[i].W_r,hptr);
				}

				hlU_rd2[i] *= Momentum_d2;
				hlU_rd2[i] += Hiddenlayers[i].U_rd2 * (1.0 - Momentum_d2);
				cuDiv(Hiddenlayers[i].lr_U, hlU_rd2[i] + mu, hidden_lr_U[i]);
				v_hl_U_r[i] *= Momentum_u;
				v_hl_U_r[i] += Hiddenlayers[i].U_rgrad.Mul(hidden_lr_U[i])
					* (1.0 - Momentum_u);
				Hiddenlayers[i].U_r -= v_hl_U_r[i];
				if(0 ==((k+1) % 300)){
					reduceMat(Hiddenlayers[i].U_r, hptr);
				}
			}
		}
	}
	trainEnd = time(NULL);
	int sec = trainEnd - trainBegin;
	int min = sec / 60;
	sec = sec % 60;
	printf("Training time : %d'%d\"\n",min,sec );
	testInit(Hiddenlayers);
	printf("Testing training data: \n");
	testNetwork(Hiddenlayers, SMR, 0);
	testInit(Hiddenlayers);
	printf("Testing test data: \n");
	testNetwork(Hiddenlayers, SMR, 1);
}

