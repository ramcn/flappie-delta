#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cblas.h>
#include <inttypes.h>

int max(int x, int y){
   if(x > y) return x;
   else return y;
}
void PrintMatrix(float* pMatrix, const size_t nR, const size_t nC, const CBLAS_ORDER Order) {
    unsigned int i, j;
    if (Order == CblasRowMajor) {
        for (i = 0; i < nR; i++) {
            for (j = 0; j < nC; j++) {
                printf("%f \t ", pMatrix[i * nC + j]); // !!!
            }
            printf("\n"); // !!!
        }
    } else {
        for (i = 0; i < nR; i++) {
            for (j = 0; j < nC; j++) {
                printf("%f \t ", pMatrix[i + j* nR ]); // !!!
            }
            printf("\n"); // !!!
        }
    }
    printf("\n"); // !!!
}

float prev_state[256];
static float acc_buff[768];
float prev_x[512];
void reset_acc_buff(){
    for(int i=0; i<768; i++)
	    acc_buff[i]=0.0;
    for(int i=0; i<256; i++)
	    prev_state[i]=0.0;
}
int skipped=0;
int flops=0;
float df;

void mat_mul_c(float *a_rm, float *b_cm, float *c_in, float *c_out,
                  uint32_t M, uint32_t N, uint32_t P,
                  bool a_trans, bool b_trans, float alpha, float beta,
                  uint32_t a_stride, uint32_t b_stride, uint32_t c_stride)
{
    uint32_t a_inner_stride, a_outer_stride, b_inner_stride, b_outer_stride;
    a_inner_stride = 1; a_outer_stride = a_stride;
    b_inner_stride = 1; b_outer_stride = b_stride;
    int ops=0;

    for(uint32_t i = 0; i < M; i++) { // M = 768
        for(uint32_t j = 0; j < P; j++) { // P = 1
            float acc = 0;
            const float *a  = a_rm + (a_outer_stride * i); // row selection
            const float *b  = b_cm + (b_inner_stride * j); // col selection
            for(uint32_t k = 0; k < N; k++) { // N = 256
                acc += a[k * a_inner_stride] * b[k * b_outer_stride];
		ops++;
            }
            uint32_t idx = (c_stride * i) + j;
            c_out[idx] = (beta * c_in[idx]) + (alpha * acc);
        }
    }
    printf("MAC ops=%d\n",ops);
}

void mat_mul_c_nodelta(float *a_rm, float *b_cm, float *c_in, float *c_out,
                  uint32_t M, uint32_t N, uint32_t P,
                  bool a_trans, bool b_trans, float alpha, float beta,
                  uint32_t a_stride, uint32_t b_stride, uint32_t c_stride, float *prev_state)
{
    uint32_t a_inner_stride, a_outer_stride, b_inner_stride, b_outer_stride;
    a_inner_stride = 1; a_outer_stride = a_stride;
    b_inner_stride = 1; b_outer_stride = b_stride;
    flops = 0; skipped=0;

    for(uint32_t i = 0; i < M; i++) { // M = 768
        for(uint32_t j = 0; j < P; j++) { // P = 1
            float acc = 0;
            const float *a  = a_rm + (a_outer_stride * i); // row selection
            const float *b  = b_cm + (b_inner_stride * j); // col selection
            uint32_t idx = (c_stride * i) + j;
            for(uint32_t k = 0; k < N; k++) { // N = 256
		float delta = b[k] - prev_state[k];
		float mul = b[k];

		if( (b[k] < 0 && b[k] > -100) || (b[k] > 0 && b[k] < 100) ) {
		}
		else if (delta < 0 && delta > -100){
			mul = prev_state[k];
			skipped++;
		}
		else if (delta > 0 && delta < 100){
			skipped++;
	 		mul = prev_state[k];
		}	
                acc += a[k * a_inner_stride] * mul;
		flops++;
            }
            c_out[idx] = (beta * c_in[idx]) + (alpha * acc);
        }
    }
}


void mat_mul_c_delta(float *a_rm, float *b_cm, float *c_in, float *c_out,
                  uint32_t M, uint32_t N, uint32_t P,
                  bool a_trans, bool b_trans, float alpha, float beta,
                  uint32_t a_stride, uint32_t b_stride, uint32_t c_stride, float *prev_state)
{
    uint32_t a_inner_stride, a_outer_stride, b_inner_stride, b_outer_stride;
    a_inner_stride = 1; a_outer_stride = a_stride;
    b_inner_stride = 1; b_outer_stride = b_stride;
    flops=0;
    skipped=0;


    for(uint32_t i = 0; i < M; i++) { // M = 768
        for(uint32_t j = 0; j < P; j++) { // P = 1
            const float *a  = a_rm + (a_outer_stride * i); // row selection
            const float *b  = b_cm + (b_inner_stride * j); // col selection
            uint32_t idx = (c_stride * i) + j;
            for(uint32_t k = 0; k < N; k++) { // N = 256
		float delta = b[k] - prev_state[k];   
		if( (b[k] < 0 && b[k] > -100) || (b[k] > 0 && b[k] < 100) ) {
                   acc_buff[idx] += a[k] * b[k];
		   flops++;
	        }
		else if (delta < 0 && delta > -100){
		   skipped++;
		}
		else if (delta > 0 && delta < 100) {
		   skipped++;
		}
		else { 
                   acc_buff[idx] += a[k] * delta;
		   flops++;
		}
                //acc_buff[idx] += a[k] * delta;
            }
	    //printf("%f ", acc_buff[idx]);
            c_out[idx] = (beta * c_in[idx]) + (alpha * acc_buff[idx]);
        }
    }	
}



int main(void)
{
    const int m = 4;
    const int n = 4;
    const int k = 1;

    //float A[] = { 8, 4, 7, 3, 5, 1, 1, 3, 2, 1, 2, 3, 2, 0, 1, 1};
    //float B[5] = { -1, 2, -1, 1 };
    float A[] = {115, 061, 110, 054, -004, 148, 032, -112, 
	         176, 047, -80, 28, -036, 138, 067, 162};

    float B[] = {062, -207,-762, 303};
    float BNEXT[] = {-233, -245, -975, -517};
    float BNEXT1[] = { 396, -020, -933, -401};

    float alpha = 1.0, beta = 0.0;
    int lda, ldb, ldc;

    float * C = (float*) malloc(m * k * sizeof(float));
    for (int i = 0; i < m*k; i++) C[i] = 0.0;

    cblas_sgemv(CblasRowMajor, CblasNoTrans, m, n, alpha, A, n, B, k, beta, C, k);
    //printf("SGEMV output\n");
    //PrintMatrix(C, m, k, CblasRowMajor);
    //mat_mul_c(A, B,C,C, m, n, 1, 0,0, alpha, beta, m, 1, 1);
    mat_mul_c_nodelta(A, B,C,C, m, n, 1, 0,0, alpha, beta, m, 1, 1, prev_state);
    //printf("non delta output\n");
    //PrintMatrix(C, m, k, CblasRowMajor);
    df = (float)skipped/(float)flops;
    //printf("no delta percentage=%f\n",df*100.0);
    //printf("total flops = %d saved flops = %d\n", flops, skipped);

    mat_mul_c_delta(A, B,C,C, m, n, 1, 0,0, alpha, beta, m, 1, 1, prev_state);
    //printf("\ndelta output\n");
    //PrintMatrix(C, m, k, CblasRowMajor);
    df = (float)skipped/(float)flops;
    //printf("delta percentage=%f\n",df*100.0);
    //printf("total flops = %d saved flops = %d\n", flops, skipped);

    for (int i = 0; i < m*k; i++) C[i] = 0.0;
    for(int i=0; i<n; i++)
	prev_state[i] = B[i];

    cblas_sgemv(CblasRowMajor, CblasNoTrans, m, n, alpha, A, n, BNEXT, k, beta, C, k);
    //printf("\nSGEMV output of BNEXT\n");
    //PrintMatrix(C, m, k, CblasRowMajor);
    mat_mul_c_nodelta(A, BNEXT,C,C, m, n, 1, 0,0, alpha, beta, m, 1, 1, prev_state);
    //printf("\nnon delta output of BNEXT\n");
    //PrintMatrix(C, m, k, CblasRowMajor);
    df = (float)skipped/(float)flops;
    //printf("no delta percentage=%f\n",df*100.0);
    //printf("total flops = %d saved flops = %d\n", flops, skipped);
    //printf("\ndelta output of BNEXT\n");
    mat_mul_c_delta(A, BNEXT,C,C, m, n, 1, 0,0, alpha, beta, m, 1, 1, prev_state);
    //PrintMatrix(C, m, k, CblasRowMajor);
    df = (float)skipped/(float)flops;
    //printf("delta percentage=%f\n",df*100.0);
    //printf("total flops = %d saved flops = %d\n", flops, skipped);


    for (int i = 0; i < m*k; i++) C[i] = 0.0;
    for(int i=0; i<n; i++)
	prev_state[i] = BNEXT[i];

    cblas_sgemv(CblasRowMajor, CblasNoTrans, m, n, alpha, A, n, BNEXT1, k, beta, C, k);
    printf("\nSGEMV output of BNEXT1\n");
    PrintMatrix(C, m, k, CblasRowMajor);
    mat_mul_c_nodelta(A, BNEXT1,C,C, m, n, 1, 0,0, alpha, beta, m, 1, 1, prev_state);
    printf("\nnon delta output of BNEXT1\n");
    PrintMatrix(C, m, k, CblasRowMajor);
    df = (float)skipped/(float)flops;
    printf("no delta percentage=%f\n",df*100.0);
    printf("total flops = %d saved flops = %d\n", flops, skipped);
    printf("\ndelta output of BNEXT1\n");
    mat_mul_c_delta(A, BNEXT1,C,C, m, n, 1, 0,0, alpha, beta, m, 1, 1, prev_state);
    PrintMatrix(C, m, k, CblasRowMajor);
    df = (float)skipped/(float)flops;
    printf("delta percentage=%f\n",df*100.0);
    printf("total flops = %d saved flops = %d\n", flops, skipped);


    free(C);

    return 0;
}
