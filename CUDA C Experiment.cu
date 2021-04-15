//Impour CUDA and nvidia graph libraries
#include <stdio.h>
#include <cuda_runtime.h>
#include <nvgraph.h>
#include <iostream>
#include <time.h>
#include <algorithm>
#include <string>
#include "C:\gurobi902\win64\include\gurobi_c.h"

//Function to check status messages
void check(nvgraphStatus_t status) {
	
	//Only check unsuccessful status
    if (status != NVGRAPH_STATUS_SUCCESS) {
        printf("ERROR : %d\n",status);
		//Exit program with 0
		//Successful exexution will return 1
        exit(0);
    }
}

//Function to generate combinations
void comb(int N, int K, int* result)
{
    std::string bitmask(K, 1); // K leading 1's
    bitmask.resize(N, 0); 	   // N-K trailing 0's
 
    // print integers and permute bitmask
	int c = 0;
    do {
        for (int i = 0; i < N; ++i) // [0..N-1] integers
        {
            //if (bitmask[i]) std::cout << " " << i;
			if (bitmask[i]) {result[c] = i;c++;}
        }
        //std::cout << std::endl;
    } while (std::prev_permutation(bitmask.begin(), bitmask.end()));
}

//Get cost function
__global__ void getCost(float* dm, int* cm, int* drivers, int* orders)
{
	unsigned long d = blockIdx.x;
	unsigned long o = threadIdx.x;
	
	cm[d*100+o] = dm[drivers[d]*1000+orders[o*2]] + dm[orders[o*2]*1000+orders[o*2+1]];

}

//Main program to execute logic
int main(int argc, char **argv) {
	
	//Program variables
	
    const size_t n = 1000;				//Number of vertices (nodes)
	const size_t nnz = 300018;				//Number of edges
	const size_t vertex_numsets = 1;	//Data types of vertex (Only one in this case)
	const size_t edge_numsets = 1;		//Data types of edges  (Only one in this case)
    float *sssp_1_h;					//Array to store path values
    void** vertex_dim;					//Array to store vertex dimentions
	int source_vert;					//Source vertext to fill up distance matrix
	
	float *dm_h;
	int *cm_h;
	dm_h = (float *)malloc(1000*1000*sizeof(float));
	cm_h = (int *)malloc(100*100*sizeof(int));
	
    // nvgraph variables
    // nvgraphStatus_t status;					//status variable to store response of nvgraph function
	nvgraphHandle_t handle;						//graph handle
    nvgraphGraphDescr_t graph;					//graph descriptor
    nvgraphCSCTopology32I_t CSC_input;			//Select Compressed Sparse Column formate
    cudaDataType_t edge_dimT = CUDA_R_32F;		//Set edge datatype float
    cudaDataType_t* vertex_dimT;				//Initialize vertax dimention
    // Init host data
    sssp_1_h = (float*)malloc(n*sizeof(float));
    vertex_dim  = (void**)malloc(vertex_numsets*sizeof(void*));
    vertex_dimT = (cudaDataType_t*)malloc(vertex_numsets*sizeof(cudaDataType_t));
    CSC_input = (nvgraphCSCTopology32I_t) malloc(sizeof(struct nvgraphCSCTopology32I_st));
    vertex_dim[0]= (void*)sssp_1_h;
	vertex_dimT[0] = CUDA_R_32F;
	
	//###########################################
	
    //float weights_h[] = {10,10,9,7,9,8,12,19,7,8,12,19};
    //int destination_offsets_h[] = {0,1,4,8,10,11,12};
    //int source_indices_h[] = {1,0,2,3,1,3,4,5,1,2,2,2};
	
	float *weights_h;
	int *destination_offsets_h;
	int *source_indices_h;
	int *drivers_h;
	int *orders_h;
	
	weights_h 				= (float *)malloc(300018 * sizeof(float));
	destination_offsets_h 	= (int   *)malloc(1001   * sizeof(int  ));
	source_indices_h		= (int   *)malloc(300018 * sizeof(int  ));
	drivers_h				= (int   *)malloc(100 * sizeof(int  ));
	orders_h				= (int   *)malloc(200 * sizeof(int  ));
	
	//Read Files
	FILE *f;
	int ntmp=0;
	
	//weights
	f = fopen("weights.csv", "r");
	if (f == NULL) {
		printf("Failed to open file\n");
	}
	while (fscanf(f, "%f", &weights_h[ntmp++]) == 1) {
		fscanf(f, ",");
	}
	fclose(f);

	//offsets
	ntmp = 0;
	f = fopen("offsets.csv", "r");
	if (f == NULL) {
		printf("Failed to open file\n");
	}
	while (fscanf(f, "%d", &destination_offsets_h[ntmp++]) == 1) {
		fscanf(f, ",");
	}
	fclose(f);
	
	//sources
	ntmp = 0;
	f = fopen("sources.csv", "r");
	if (f == NULL) {
		printf("Failed to open file\n");
	}
	while (fscanf(f, "%d", &source_indices_h[ntmp++]) == 1) {
		fscanf(f, ",");
	}
	fclose(f);
	
	//drivers
	ntmp = 0;
	f = fopen("drivers.csv", "r");
	if (f == NULL) {
		printf("Failed to open file\n");
	}
	while (fscanf(f, "%d", &drivers_h[ntmp++]) == 1) {
		fscanf(f, ",");
	}
	fclose(f);
	
	//orders
	ntmp = 0;
	f = fopen("orders.csv", "r");
	if (f == NULL) {
		printf("Failed to open file\n");
	}
	while (fscanf(f, "%d", &orders_h[ntmp++]) == 1) {
		fscanf(f, ",");
	}
	fclose(f);
	
	
	//###########################################
    check(nvgraphCreate(&handle));
    check(nvgraphCreateGraphDescr (handle, &graph));
	
    CSC_input->nvertices = n;
	CSC_input->nedges = nnz;
    CSC_input->destination_offsets = destination_offsets_h;
    CSC_input->source_indices = source_indices_h;
	
    // Set graph connectivity and properties (tranfers)
    check(nvgraphSetGraphStructure(handle, graph, (void*)CSC_input, NVGRAPH_CSC_32));
    check(nvgraphAllocateVertexData(handle, graph, vertex_numsets, vertex_dimT));
    check(nvgraphAllocateEdgeData  (handle, graph, edge_numsets, &edge_dimT));
    check(nvgraphSetEdgeData(handle, graph, (void*)weights_h, 0));
	
	//#################################################################
	
	f = fopen("result_CUDA.csv","w");

	//Check if file can be opened
	if(f == NULL)
	{
		printf("Failed to open or create file. \n");            
	}
	
	for(int i = 0; i< n; i++){
		
		// Solve
		source_vert = i;
		check(nvgraphSssp(handle, graph, 0,  &source_vert, 0));
		// Get and print result
		check(nvgraphGetVertexData(handle, graph, (void*)sssp_1_h, 0));
		
		
		for(int j = 0; j< n; j++){
			fprintf(f,"%f\n",sssp_1_h[j]);
			dm_h[i*1000+j] = sssp_1_h[j];
		}
	}
	
	fclose(f);
	
	//################################################################
	
    //Clean 
    free(sssp_1_h); free(vertex_dim);
    free(vertex_dimT); free(CSC_input);
    check(nvgraphDestroyGraphDescr(handle, graph));
    check(nvgraphDestroy(handle));
	
	//################################################################
	
	//Transfer data to GPU
	float *dm_d;
	int *drivers_d;
	int *orders_d;
	int *cm_d;
	
	cudaMalloc((void **)&dm_d, 1000 * 1000	* sizeof(float));
	cudaMalloc((void **)&drivers_d, 100 * sizeof(int));
	cudaMalloc((void **)&orders_d, 200 * sizeof(int));
	cudaMalloc((void **)&cm_d, 100 * 100 * sizeof(int));
	
	cudaMemcpy(dm_d	 		 , dm_h				, 1000	 * 1000	* sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(drivers_d	 , drivers_h		, 100	 * 1	* sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(orders_d		 , orders_h			, 200	 * 1	* sizeof(int), cudaMemcpyHostToDevice);

	printf("\n");
	
	//Initialize start time
	clock_t startTime = clock();
	
	getCost<<<100,100>>>(dm_d,cm_d,drivers_d,orders_d);

	cudaMemcpy(cm_h		 , cm_d			, 100	 * 100	* sizeof(int), cudaMemcpyDeviceToHost);
	
	for(int i=0;i<100;i++)
	{
		for(int j=0;j<100;j++)
		{
			printf("%d ",cm_h[i*100+j]);
		}
		printf("\n");
	}
	
	//Initialize end time
	clock_t endTime = clock();
	
	//#######################################################################
	
	GRBenv   *env   = NULL;
	GRBmodel *model = NULL;
	int       error = 0;
	double    sol[10000];
	int       ind[10000];
	double    val[10000];
	double    obj[10000];
	char      vtype[10000];
	int       optimstatus;
	double    objval;

	// Create environment
	error = GRBemptyenv(&env);
	if (error) goto QUIT;

	// Set log files
	error = GRBsetstrparam(env, "LogFile", "mip1.log");
	if (error) goto QUIT;

	// Start environment
	error = GRBstartenv(env);
	if (error) goto QUIT;

	// Create an empty model
	error = GRBnewmodel(env, &model, "mip1", 0, NULL, NULL, NULL, NULL, NULL);
	if (error) goto QUIT;

	// Add variables
	
	for(int i=0; i<10000; i++)
	{
		obj[i] = cm_h[i];
	}
	
	for(int i=0; i<10000; i++)
	{
		vtype[i] = GRB_BINARY;
	}
	
	error = GRBaddvars(model, 10000, 0, NULL, NULL, NULL, obj, NULL, NULL, vtype, NULL);
	if (error) goto QUIT;

	/* Change objective sense to maximization
	error = GRBsetintattr(model, GRB_INT_ATTR_MODELSENSE, GRB_MAXIMIZE);
	if (error) goto QUIT; */

	//set positions before constraints
	for(int i=0; i<10000; i++)
	{
		ind[i] = i;
	}

	// Constraint: sum(rows) <= 1
	
	printf("Adding rows\n");
	
	for(int i=0; i<100; i++)
	{
		for(int j=0; j<10000; j++)
		{
			val[j] = 0;
		}
		
		for(int j=0; j<100; j++)
		{
			val[i*100 + j] = 1;
		}
		
		error = GRBaddconstr(model, 10000, ind, val, GRB_LESS_EQUAL, 1.0, "cr");
		if (error) goto QUIT;
	}
	
	// Constraint: sum(cols) <= 1
	
	printf("Adding cols\n");
	
	for(int i=0; i<100; i++)
	{
		for(int j=0; j<10000; j++)
		{
			val[j] = 0;
		}
		
		for(int j=0; j<100; j++)
		{
			val[j*100 + i] = 1;
		}
		
		error = GRBaddconstr(model, 10000, ind, val, GRB_LESS_EQUAL, 1.0, "cc");
		if (error) goto QUIT;
	}
	
	// Constraint: min assignment = 100
	
	printf("Adding limit\n");
	
	for(int i=0; i<10000; i++)
	{
		val[i] = 1;
	}
	
	error = GRBaddconstr(model, 10000, ind, val, GRB_EQUAL, 100.0, "cl");
	if (error) goto QUIT;
	
	printf("Adding done\n");

	// Optimize model
	error = GRBoptimize(model);
	if (error) goto QUIT;

	// Write model to 'mip1.lp'
	error = GRBwrite(model, "mip1.lp");
	if (error) goto QUIT;

	// Get solution information
	error = GRBgetintattr(model, GRB_INT_ATTR_STATUS, &optimstatus);
	if (error) goto QUIT;

	error = GRBgetdblattr(model, GRB_DBL_ATTR_OBJVAL, &objval);
	if (error) goto QUIT;

	error = GRBgetdblattrarray(model, GRB_DBL_ATTR_X, 0, 10000, sol);
	if (error) goto QUIT;

	printf("\nOptimization complete\n");
	if (optimstatus == GRB_OPTIMAL)
	{
		printf("Optimal objective: %.4e\n", objval);

		for(int i=0;i<100;i++)
		{
			for(int j=0;j<100;j++)
			{
				printf("%d ",int(sol[i*100+j]));
			}
			printf("\n");
		}
		
		printf("\n\nSelected Vals:\n");
		
		for(int i=0;i<100;i++)
		{
			for(int j=0;j<100;j++)
			{
				printf("%d ",int(sol[i*100+j])*int(cm_h[i*100+j]));
			}
			printf("\n");
		}
	}
	else if (optimstatus == GRB_INF_OR_UNBD)
	{
		printf("Model is infeasible or unbounded\n");
	}
	else
	{
		printf("Optimization was stopped early\n");
	}
	
	//Print time taken by the operation
	printf("Elapsed: %f seconds\n", (double)(endTime - startTime) / CLOCKS_PER_SEC);

QUIT:

	/* Error reporting */
	if (error)
	{
		printf("ERROR: %s\n", GRBgeterrormsg(env));
		exit(1);
	}

	/* Free model */
	GRBfreemodel(model);

	/* Free environment */
	GRBfreeenv(env);

    return 0;
}
