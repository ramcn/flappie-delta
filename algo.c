a: weight; b:state; cin: xvector; act(cout):ostate

for(idx = 0; idx < M; idx++) { // M = 768
   acc = 0;
   for(uint32_t k = 0; k < N; k++) { // N = 256
       acc += a[k] * b[k];
   }
   c_out[idx] = c_in[idx] + acc;

for(idx = 0; idx < M; idx++) { // M = 768
   for(uint32_t k = 0; k < N; k++) { // N = 256
	float delta = b[k] - prev_state[k];   
	if (delta < 0 && delta > -0.1)
	   skipped++; total_flops++;
	if (delta > 0 && delta < 0.1) 
   	   skipped++; total_flops++;
	else  
           acc_buff[idx] += a[k] * delta;
	   flops++;	
	   total_flops++;
   c_out[idx] = c_in[idx] + acc_buff[idx];

