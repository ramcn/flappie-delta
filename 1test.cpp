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
                acc += a[k * a_inner_stride] * b[k];
		flops++;
            }
	    //printf("%f ",acc);
            c_out[idx] = (beta * c_in[idx]) + (alpha * acc);
        }
    }
    //printf("\n");
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
		if( (b[k] < 0 && b[k] > -0.1) || (b[k] > 0 && b[k] < 0.1) ) {
                   acc_buff[idx] += a[k] * delta;
		   flops++;
	        }
		else if (delta < 0 && delta > -0.1){
		   skipped++;
		}
		else if (delta > 0 && delta < 0.1) {
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
    float A[] = { 0.115273811, 0.0612734258, 0.110963464, 0.0549703576, -0.00443203887, 0.148828268, 0.0321667567, -0.112016007, 
	          0.176474452, 0.0472295098, -0.080984734, 0.0283038579, -0.0364438482, 0.138353914, 0.0674756244, 0.162208289};
    float B[] = {0.062466, -0.207845,-0.762880, 0.303334};
    float BNEXT[] = {-0.233847, -0.245421, -0.975424, -0.517330};

    float alpha = 1.0, beta = 0.0;
    int lda, ldb, ldc;

    float * C = (float*) malloc(m * k * sizeof(float));
    for (int i = 0; i < m*k; i++) C[i] = 0.0;

    cblas_sgemv(CblasRowMajor, CblasNoTrans, m, n, alpha, A, n, B, k, beta, C, k);
    printf("SGEMV output\n");
    PrintMatrix(C, m, k, CblasRowMajor);
    //mat_mul_c(A, B,C,C, m, n, 1, 0,0, alpha, beta, m, 1, 1);
    mat_mul_c_nodelta(A, B,C,C, m, n, 1, 0,0, alpha, beta, m, 1, 1, prev_state);
    printf("non delta output\n");
    PrintMatrix(C, m, k, CblasRowMajor);
    df = (float)skipped/(float)flops;
    printf("no delta percentage=%f\n",df*100.0);
    printf("total flops = %d saved flops = %d\n", flops, skipped);

    mat_mul_c_delta(A, B,C,C, m, n, 1, 0,0, alpha, beta, m, 1, 1, prev_state);
    printf("\ndelta output\n");
    PrintMatrix(C, m, k, CblasRowMajor);
    df = (float)skipped/(float)flops;
    printf("delta percentage=%f\n",df*100.0);
    printf("total flops = %d saved flops = %d\n", flops, skipped);

    for (int i = 0; i < m*k; i++) C[i] = 0.0;

    cblas_sgemv(CblasRowMajor, CblasNoTrans, m, n, alpha, A, n, BNEXT, k, beta, C, k);
    printf("\nSGEMV output of BNEXT\n");
    PrintMatrix(C, m, k, CblasRowMajor);
    mat_mul_c_nodelta(A, BNEXT,C,C, m, n, 1, 0,0, alpha, beta, m, 1, 1, prev_state);
    printf("\nnon delta output of BNEXT\n");
    PrintMatrix(C, m, k, CblasRowMajor);
    df = (float)skipped/(float)flops;
    printf("no delta percentage=%f\n",df*100.0);
    printf("total flops = %d saved flops = %d\n", flops, skipped);

    for(int i=0; i<n; i++)
	prev_state[i] = B[i];

    printf("\ndelta output of BNEXT\n");
    mat_mul_c_delta(A, BNEXT,C,C, m, n, 1, 0,0, alpha, beta, m, 1, 1, prev_state);
    PrintMatrix(C, m, k, CblasRowMajor);
    df = (float)skipped/(float)flops;
    printf("delta percentage=%f\n",df*100.0);
    printf("total flops = %d saved flops = %d\n", flops, skipped);


    free(C);

    return 0;
}
