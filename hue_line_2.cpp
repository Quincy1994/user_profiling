#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <gsl/gsl_rng.h>
#include <vector>
using namespace std;

#define MAX_STRING 100
#define SIGMOID_BOUND 6
#define NEG_SAMPLING_POWER 0.75

#define random(x) (rand()%x)

const int hash_table_size = 100000000;
const int neg_table_size = 1e8;
const int sigmoid_table_size = 1000;

typedef float real;                    // Precision of float numbers


struct ClassVertex {
	double mw_degree;
	double um_degree;
	double uw_degree;
	char *name;
};



// loading network_file
char message_word_file[MAX_STRING], user_word_file[MAX_STRING], user_message_file[MAX_STRING];
// output embedding file
char embedding_file[MAX_STRING], context_file[MAX_STRING];

// define vertex
struct ClassVertex *vertex;
int max_num_vertices = 1000, num_vertices = 0;
int *vertex_hash_table;

int is_binary = 0, num_threads = 1, order = 2, dim = 100, num_negative = 5;

long long mw_current_sample_count = 0, uw_current_sample_count = 0, um_current_sample_count = 0;
long long mw_total_samples = 1, uw_total_samples = 1, um_total_samples = 1;
real init_rho = 0.025, rho;
real *emb_vertex, *emb_context, *sigmoid_table;

// edge weight
int *mw_edge_source_id, *mw_edge_targe_id;
int *uw_edge_source_id, *uw_edge_target_id;
int *um_edge_source_id, *um_edge_target_id;
double *mw_edge_weight, *uw_edge_weight, *um_edge_weight;


//edges
long long mw_num_edges = 0;
long long uw_num_edges = 0;
long long um_num_edges = 0;

// Parameters for edge sampling
long long *mw_alias, double *mw_prob;
long long *uw_alias, double *uw_prob;
long long *um_alias, double *um_prob;

// negative table
int *mw_neg_table, *uw_neg_table, *um_neg_table;

// loss paramters;
int iteration = 20;

const gsl_rng_type * gsl_T;
gsl_rng * gsl_r;

/* Build a hash table, mapping each vertex name to a unique vertex id */
unsigned int Hash(char *key)
{
	unsigned int seed = 131;
	unsigned int hash = 0;
	while (*key)
	{
		hash = hash * seed + (*key++);
	}
	return hash % hash_table_size;
}

void InitHashTable()
{
	vertex_hash_table = (int *)malloc(hash_table_size * sizeof(int));
	for (int k = 0; k != hash_table_size; k++) vertex_hash_table[k] = -1;
}

void InsertHashTable(char *key, int value)
{
	int addr = Hash(key);
	while (vertex_hash_table[addr] != -1) addr = (addr + 1) % hash_table_size;
	vertex_hash_table[addr] = value;
}

int SearchHashTable(char *key)
{
	int addr = Hash(key);
	while(1)
	{
		if (vertex_hash_table[addr] == -1) return -1;
		if (!strcmp(key, vertex[vertex_hash_table[addr]].name)) return vertex_hash_table[addr];
		addr = (addr + 1) % hash_table_size;
	}
	return -1;
}

/* Add a vertex to the vertex set */
int AddVertex(char *name)
{
	int length = strlen(name) + 1;
	if (length > MAX_STRING) length = MAX_STRING;
	vertex[num_vertices].name = (char *)calloc(length, sizeof(char));
	strncpy(vertex[num_vertices].name, name, length-1);
	vertex[num_vertices].mw_degree = 0;
	vertex[num_vertices].um_degree = 0;
	vertex[num_vertices].uw_degree = 0;

	num_vertices++;
	if (num_vertices + 2 >= max_num_vertices)
	{
		max_num_vertices += 1000;
		vertex = (struct ClassVertex *)realloc(vertex, max_num_vertices * sizeof(struct ClassVertex));
	}
	InsertHashTable(name, num_vertices - 1);
	return num_vertices - 1;
}

// message_word_file[MAX_STRING], user_word_file[MAX_STRING], user_message_file[MAX_STRING];
// Read vectex from the training file
void LoadVertex()
{
	FILE *fin;
	char name_v1[MAX_STRING], name_v2[MAX_STRING], str[2 * MAX_STRING + 10000];
	int vid_1, vid_2;
	double weight;
	int num_edges = 0;
	num_vertices = 0;

	// message word file
	fin = fopen(message_word_file, "rb");
	if (fin == NULL)
	{
		printf("Error: message word file not found\n");
		exit(1);
	}
	while(fgets(str, sizeof(str), fin)) num_edges++;
	fclose(fin);
	fin = fopen(message_word_file, "rb");
	for (int k = 0; k != num_edges; k++)
	{
		fscanf(fin, "%s %s %lf", name_v1, name_v2, &weight);

		if (k % 10000 == 0)
		{
			printf("Reading message word edges: %.3lf%%%c", k / (double)(num_edges + 1) * 100, 13);
			fflush(stdout);
		}
		vid_1 = SearchHashTable(name_v1);
		if (vid_1 == -1) vid_1 = AddVertex(name_v1);
		vertex[vid_1].mw_degree += weight;

		vid_2 = SearchHashTable(name_v2);
		if (vid_2 == -1) vid_2 = AddVertex(name_v2);
		vertex[vid_2].mw_degree += weight;

	}
	fclose(fin);
	printf("Number of message word vertices: %d          \n", num_vertices);

	// user word file
    num_edges = 0;
    fin = fopen(user_word_file, "rb");
    if (fin == NULL)
    {
        print("Error: user word file not found\n");
        exit(1);
    }
    while(fgets(str, sizeof(str), fin)) num_edges++;
    fclose(fin);
    fin = fopen(user_word_file, "rb");
    for (int k = 0; k != num_edges; k++){
        fscanf(fin, "%s %s %lf", name_v1, name_v2, &weight);
        if (k % 10000 == 0){
            printf("Reading user word edges: %3.lf%%%c", k / (double)(num_edges + 1) * 100, 13);
            fflush(stdout);
        }
        vid_1 = SearchHashTable(name_v1);
        if (vid_1 == -1) vid_1 = AddVertex(name_v1);
        vertex[vid_1].uw_degree += weight;

        vid_2 = SearchHashTable(name_v2);
        if (vid_2 == -1) vid_2 = AddVertex(name_v2);
        vertex[vid_2].uw_degree += weight;
    }
    fclose(fin);
    print("Number of user word vertices: %d         \n", num_vertices);

    // user message file
    num_edges = 0;
    fin = open(user_message_file, "rb");
    if (fin == NULL)
    {
        print("Error: user message file not found\n");
        exit(1);
    }
    while(fgets(str, sizeof(str), fin)) num_edges++;
    fclose(fin);
    fin = fopen(user_message_file, "rb");
    for(int k = 0; k != num_edges; k++){

        fscanf(fin, "%s %s %lf", name_v1, name_v2, &weight);

        if(k % 10000 == 0){
            printf("Reading user message edges: %3.lf%%%c", k / (double)(num_edges + 1) * 100, 13);
            fflush(stdout);
        }
        vid_1 = SearchHashTable(name_v1);
        if (vid_1 == -1) vid_1 = AddVertex(name_v1);
        vertex[vid_1].um_degree += weight;

        vid_2 = SearchHashTable(name_v2);
        if (vid_2 == -1) vid_2 = AddVertex(name_v2);
        vertex[vid_2].um_degree += weight;
    }
    fclose(fin);
    print("Number of user message vertices: %d          \n", num_vertices);
}

// Read edges
void LoadEdges()
{
	FILE *fin;
	char name_v1[MAX_STRING], name_v2[MAX_STRING], str[2 * MAX_STRING + 10000];
	int vid;
	double weight;

	// message word
	fin = fopen(message_word_file, "rb");
	if (fin == NULL)
	{
		printf("ERROR: message word file not found!\n");
		exit(1);
	}
	mw_num_edges = 0;
	while(fgets(str, sizeof(str), fin)) {
		mw_num_edges++;
	}
	fclose(fin);
	printf("Number of message word edges: %lld		\n", mw_num_edges++);

	mw_edge_source_id = (int *)malloc(mw_num_edges * sizeof(int));
	mw_edge_target_id = (int *)malloc(mw_num_edges * sizeof(int));
	mw_edge_weight = (double *)malloc(mw_num_edges * sizeof(double));
	if (mw_edge_source_id == NULL || mw_edge_target_id == NULL || mw_edge_weight == NULL)
	{
		printf("Error: memory allocation failed!\n");
		exit(1);
	}

	// read message word edges;
	fin = fopen(message_word_file, "rb");
	for(int k = 0; k != mw_num_edges; k++)
	{
		fscanf(fin, "%s %s %lf", name_v1, name_v2, &weight);

		if (k % 10000 == 0)
		{
			printf("Reading message word edges: %.3lf%%%c", k / (double)(mw_num_edges + 1) * 100, 13);
			fflush(stdout);
		}

		vid = SearchHashTable(name_v1);
		mw_edge_source_id[k] = vid;
		vid = SearchHashTable(name_v2);
		mw_edge_target_id[k] = vid;
		mw_edge_weight[k] = weight;
	}
	fclose(fin);

	// user word
	fin = fopen(user_word_file, "rb");
	if (fin == NULL)
	{
		printf("ERROR: user word file not found!\n");
		exit(1);
	}
	uw_num_edges = 0;
	while(fgets(str, sizeof(str), fin)) {
		uw_num_edges++;
	}
	fclose(fin);
	printf("Number of user word edges: %lld		\n", uw_num_edges++);

	uw_edge_source_id = (int *)malloc(uw_num_edges * sizeof(int));
	uw_edge_target_id = (int *)malloc(uw_num_edges * sizeof(int));
	uw_edge_weight = (double *)malloc(uw_num_edges * sizeof(double));
	if (uw_edge_source_id == NULL || uw_edge_target_id == NULL || uw_edge_weight == NULL)
	{
		printf("Error: memory allocation failed!\n");
		exit(1);
	}

	// read message word edges;
	fin = fopen(user_word_file, "rb");
	for(int k = 0; k != uw_num_edges; k++)
	{
		fscanf(fin, "%s %s %lf", name_v1, name_v2, &weight);

		if (k % 10000 == 0)
		{
			printf("Reading user word edges: %.3lf%%%c", k / (double)(uw_num_edges + 1) * 100, 13);
			fflush(stdout);
		}

		vid = SearchHashTable(name_v1);
		uw_edge_source_id[k] = vid;
		vid = SearchHashTable(name_v2);
		uw_edge_target_id[k] = vid;
		uw_edge_weight[k] = weight;
	}
	fclose(fin);

	// user message
	fin = fopen(user_message_file, "rb");
	if (fin == NULL)
	{
		printf("ERROR: user message file not found!\n");
		exit(1);
	}
	um_num_edges = 0;
	while(fgets(str, sizeof(str), fin)) {
		um_num_edges++;
	}
	fclose(fin);
	printf("Number of user message edges: %lld		\n", um_num_edges++);

	um_edge_source_id = (int *)malloc(um_num_edges * sizeof(int));
	um_edge_target_id = (int *)malloc(um_num_edges * sizeof(int));
	um_edge_weight = (double *)malloc(um_num_edges * sizeof(double));
	if (um_edge_source_id == NULL || um_edge_target_id == NULL || um_edge_weight == NULL)
	{
		printf("Error: memory allocation failed!\n");
		exit(1);
	}

	// read user message edges;
	fin = fopen(user_message_file, "rb");
	for(int k = 0; k != um_num_edges; k++)
	{
		fscanf(fin, "%s %s %lf", name_v1, name_v2, &weight);

		if (k % 10000 == 0)
		{
			printf("Reading user message edges: %.3lf%%%c", k / (double)(um_num_edges + 1) * 100, 13);
			fflush(stdout);
		}

		vid = SearchHashTable(name_v1);
		um_edge_source_id[k] = vid;
		vid = SearchHashTable(name_v2);
		um_edge_target_id[k] = vid;
		um_edge_weight[k] = weight;
	}
	fclose(fin);
	printf("Edges have been loaded.");
}


/* The alias sampling algorithm, which is used to sample an edge in O(1) time. */
void InitMWAliasTable()
{
	mw_alias = (long long *)malloc(mw_num_edges*sizeof(long long));
	mw_prob = (double *)malloc(mw_num_edges*sizeof(double));
	if (mw_alias == NULL || mw_prob == NULL)
	{
		printf("Error: memory allocation failed!\n");
		exit(1);
	}

	double *norm_prob = (double*)malloc(mw_num_edges*sizeof(double));
	long long *large_block = (long long*)malloc(mw_num_edges*sizeof(long long));
	long long *small_block = (long long*)malloc(mw_num_edges*sizeof(long long));
	if (norm_prob == NULL || large_block == NULL || small_block == NULL)
	{
		printf("Error: memory allocation failed!\n");
		exit(1);
	}

	double sum = 0;
	long long cur_small_block, cur_large_block;
	long long num_small_block = 0, num_large_block = 0;

	for (long long k = 0; k != mw_num_edges; k++) sum += mw_edge_weight[k];
	for (long long k = 0; k != mw_num_edges; k++) norm_prob[k] = mw_edge_weight[k] * mw_num_edges / sum;

	for (long long k = mw_num_edges - 1; k >= 0; k--)
	{
		if (norm_prob[k]<1)
			small_block[num_small_block++] = k;
		else
			large_block[num_large_block++] = k;
	}

	while (num_small_block && num_large_block)
	{
		cur_small_block = small_block[--num_small_block];
		cur_large_block = large_block[--num_large_block];
		mw_prob[cur_small_block] = norm_prob[cur_small_block];
		mw_alias[cur_small_block] = cur_large_block;
		norm_prob[cur_large_block] = norm_prob[cur_large_block] + norm_prob[cur_small_block] - 1;
		if (norm_prob[cur_large_block] < 1)
			small_block[num_small_block++] = cur_large_block;
		else
			large_block[num_large_block++] = cur_large_block;
	}

	while (num_large_block) mw_prob[large_block[--num_large_block]] = 1;
	while (num_small_block) mw_prob[small_block[--num_small_block]] = 1;

	free(norm_prob);
	free(small_block);
	free(large_block);
}

void InitUWAliasTable()
{
	uw_alias = (long long *)malloc(uw_num_edges*sizeof(long long));
	uw_prob = (double *)malloc(uw_num_edges*sizeof(double));
	if (uw_alias == NULL || uw_prob == NULL)
	{
		printf("Error: memory allocation failed!\n");
		exit(1);
	}

	double *norm_prob = (double*)malloc(uw_num_edges*sizeof(double));
	long long *large_block = (long long*)malloc(uw_num_edges*sizeof(long long));
	long long *small_block = (long long*)malloc(uw_num_edges*sizeof(long long));
	if (norm_prob == NULL || large_block == NULL || small_block == NULL)
	{
		printf("Error: memory allocation failed!\n");
		exit(1);
	}

	double sum = 0;
	long long cur_small_block, cur_large_block;
	long long num_small_block = 0, num_large_block = 0;

	for (long long k = 0; k != uw_num_edges; k++) sum += uw_edge_weight[k];
	for (long long k = 0; k != uw_num_edges; k++) norm_prob[k] = uw_edge_weight[k] * uw_num_edges / sum;

	for (long long k = uw_num_edges - 1; k >= 0; k--)
	{
		if (norm_prob[k]<1)
			small_block[num_small_block++] = k;
		else
			large_block[num_large_block++] = k;
	}

	while (num_small_block && num_large_block)
	{
		cur_small_block = small_block[--num_small_block];
		cur_large_block = large_block[--num_large_block];
		uw_prob[cur_small_block] = norm_prob[cur_small_block];
		uw_alias[cur_small_block] = cur_large_block;
		norm_prob[cur_large_block] = norm_prob[cur_large_block] + norm_prob[cur_small_block] - 1;
		if (norm_prob[cur_large_block] < 1)
			small_block[num_small_block++] = cur_large_block;
		else
			large_block[num_large_block++] = cur_large_block;
	}

	while (num_large_block) uw_prob[large_block[--num_large_block]] = 1;
	while (num_small_block) uw_prob[small_block[--num_small_block]] = 1;

	free(norm_prob);
	free(small_block);
	free(large_block);
}

void InitUMAliasTable()
{
	um_alias = (long long *)malloc(um_num_edges*sizeof(long long));
	um_prob = (double *)malloc(um_num_edges*sizeof(double));
	if (um_alias == NULL || um_prob == NULL)
	{
		printf("Error: memory allocation failed!\n");
		exit(1);
	}

	double *norm_prob = (double*)malloc(um_num_edges*sizeof(double));
	long long *large_block = (long long*)malloc(um_num_edges*sizeof(long long));
	long long *small_block = (long long*)malloc(um_num_edges*sizeof(long long));
	if (norm_prob == NULL || large_block == NULL || small_block == NULL)
	{
		printf("Error: memory allocation failed!\n");
		exit(1);
	}

	double sum = 0;
	long long cur_small_block, cur_large_block;
	long long num_small_block = 0, num_large_block = 0;

	for (long long k = 0; k != um_num_edges; k++) sum += um_edge_weight[k];
	for (long long k = 0; k != um_num_edges; k++) norm_prob[k] = um_edge_weight[k] * um_num_edges / sum;

	for (long long k = um_num_edges - 1; k >= 0; k--)
	{
		if (norm_prob[k]<1)
			small_block[num_small_block++] = k;
		else
			large_block[num_large_block++] = k;
	}

	while (num_small_block && num_large_block)
	{
		cur_small_block = small_block[--num_small_block];
		cur_large_block = large_block[--num_large_block];
		um_prob[cur_small_block] = norm_prob[cur_small_block];
		um_alias[cur_small_block] = cur_large_block;
		norm_prob[cur_large_block] = norm_prob[cur_large_block] + norm_prob[cur_small_block] - 1;
		if (norm_prob[cur_large_block] < 1)
			small_block[num_small_block++] = cur_large_block;
		else
			large_block[num_large_block++] = cur_large_block;
	}

	while (num_large_block) um_prob[large_block[--num_large_block]] = 1;
	while (num_small_block) um_prob[small_block[--num_small_block]] = 1;

	free(norm_prob);
	free(small_block);
	free(large_block);
}


long long SampleMWAnEdge(double rand_value1, double rand_value2)
{
	long long k = (long long)mw_num_edges * rand_value1;
	return rand_value2 < mw_prob[k] ? k : mw_alias[k];
}
long long SampleUWAnEdge(double rand_value1, double rand_value2)
{
	long long k = (long long)uw_num_edges * rand_value1;
	return rand_value2 < uw_prob[k] ? k : uw_alias[k];
}
long long SampleUMAnEdge(double rand_value1, double rand_value2)
{
	long long k = (long long)um_num_edges * rand_value1;
	return rand_value2 < um_prob[k] ? k : um_alias[k];
}


/* Initialize the vertex embedding and the context embedding */
void InitVector()
{
	long long a, b;

	a = posix_memalign((void **)&emb_vertex, 128, (long long)num_vertices * dim * sizeof(real));
	if (emb_vertex == NULL) { printf("Error: memory allocation failed\n"); exit(1); }
	for (b = 0; b < dim; b++) for (a = 0; a < num_vertices; a++)
		emb_vertex[a * dim + b] = (rand() / (real)RAND_MAX - 0.5) / dim; // notice the effect of init parameter

	a = posix_memalign((void **)&emb_context, 128, (long long)num_vertices * dim * sizeof(real));
	if (emb_context == NULL) { printf("Error: memory allocation failed\n"); exit(1); }
	for (b = 0; b < dim; b++) for (a = 0; a < num_vertices; a++)
		emb_context[a * dim + b] = (rand() / (real)RAND_MAX - 0.5) / dim;
}

/* Sample negative vertex samples according to vertex degrees */
void InitMWNegTable()
{
    char begin = 'W';
    double sum = 0, cur_sum = 0, por = 0;
	int vid = 0;
	mw_neg_table = (int *)malloc(neg_table_size * sizeof(int));
	for (int k = 0; k != num_vertices; k++){
        if(vertex[k].name[0] == begin){
            sum += pow(vertex[k].mw_degree, NEG_SAMPLING_POWER);
        }
	}
	printf("sum: %lf", sum);
	int word_num = 0;
	for (int k = 0; k != neg_table_size; k++)
	{
		if ((double)(k + 1) / neg_table_size > por)
		{
		    while(1){
		        if(vertex[vid].name[0] == begin){
		            break;
		        }
		        vid++;
		    }
			cur_sum += pow(vertex[vid].mw_degree, NEG_SAMPLING_POWER);
			por = cur_sum / sum;
			vid++;
			word_num++;
		}
		mw_neg_table[k] = vid - 1;
	}
	printf("mw word num %d\n", word_num);
}

void InitUWNegTable()
{
    char begin = 'W';
	double sum = 0, cur_sum = 0, por = 0;
	int vid = 0;
	uw_neg_table = (int *)malloc(neg_table_size * sizeof(int));
	for (int k = 0; k != num_vertices; k++){
	if(vertex[k].name[0] == begin){
        sum += pow(vertex[k].uw_degree, NEG_SAMPLING_POWER);
        }
    }
	printf("sum: %lf", sum);
	int word_num = 0;
	for (int k = 0; k != neg_table_size; k++)
	{
		if ((double)(k + 1) / neg_table_size > por)
		{
		    while(1){
		        if(vertex[vid].name[0] == begin){
		            break;
		        }
		        vid = vid + 1;
		    }
			cur_sum += pow(vertex[vid].uw_degree, NEG_SAMPLING_POWER);
			por = cur_sum / sum;
			vid++;
			word_num++;
		}
		uw_neg_table[k] = vid - 1;
	}
	printf("uw word num %d\n", word_num);
}

void InitUMNegTable()
{
    char begin = 'M';
	double sum = 0, cur_sum = 0, por = 0;
	int vid = 0;
	um_neg_table = (int *)malloc(neg_table_size * sizeof(int));
	for (int k = 0; k != num_vertices; k++){
        if(vertex[k].name[0] == begin){
            sum += pow(vertex[k].um_degree, NEG_SAMPLING_POWER);
        }
	}
	printf("sum: %lf", sum);
	int message_num = 0;
	for (int k = 0; k != neg_table_size; k++)
	{
		if ((double)(k + 1) / neg_table_size > por)
		{
		    while(1){
		        if(vertex[vid].name[0] == begin){
		            break;
		        }
		        vid = vid + 1;
		    }
			cur_sum += pow(vertex[vid].um_degree, NEG_SAMPLING_POWER);
			por = cur_sum / sum;
			vid++;
			message_num++;
		}
		um_neg_table[k] = vid - 1;
	}
	printf("um word num %d\n", message_num);
}


/* Fastly compute sigmoid function */
void InitSigmoidTable()
{
	real x;
	sigmoid_table = (real *)malloc((sigmoid_table_size + 1) * sizeof(real));
	for (int k = 0; k != sigmoid_table_size; k++)
	{
		x = 2 * SIGMOID_BOUND * k / sigmoid_table_size - SIGMOID_BOUND;
		sigmoid_table[k] = 1 / (1 + exp(-x));
	}
}

real FastSigmoid(real x)
{
	if (x > SIGMOID_BOUND) return 1;
	else if (x < -SIGMOID_BOUND) return 0;
	int k = (x + SIGMOID_BOUND) * sigmoid_table_size / SIGMOID_BOUND / 2;
	return sigmoid_table[k];
}

/* Fastly generate a random integer */
int Rand(unsigned long long &seed)
{
	seed = seed * 25214903917 + 11;
	return (seed >> 16) % neg_table_size;
}

/* Update embeddings */
void Update(real *vec_u, real *vec_v, real *vec_error, int label)
{
	real x = 0, g;
	for (int c = 0; c != dim; c++) x += vec_u[c] * vec_v[c];
	g = (label - FastSigmoid(x)) * rho;
	for (int c = 0; c != dim; c++) vec_error[c] += g * vec_v[c];
	for (int c = 0; c != dim; c++) vec_v[c] += g * vec_u[c];
}

void *TrainMWLINEThread(void *id)
{
	long long m, w, lm, lw, target, label;
	long long count = 0, last_count = 0, curedge;
	unsigned long long seed = (long long)id;
	real *vec_error = (real *)calloc(dim, sizeof(real));


	while (1)
	{
		//judge for exit
		if (count > mw_total_samples / num_threads + 2) break;

		if (count - last_count > 10000)
		{
			mw_current_sample_count += count - last_count;
			last_count = count;
			printf("%cRho: %f  train MW Progress: %.3lf%%", 13, rho, (real)mw_current_sample_count / (real)(mw_total_samples + 1) * 100);
			fflush(stdout);
//			rho = init_rho * (1 - current_sample_count / (real)(total_samples + 1));
//			if (rho < init_rho * 0.0001) rho = init_rho * 0.0001;
		}

        curedge = SampleMWAnEdge(gsl_rng_uniform(gsl_r), gsl_rng_uniform(gsl_r));
        m = mw_edge_source_id[curedge];
        w = mw_edge_target_id[curedge];

        lm = m * dim;
        for (int c = 0; c != dim; c++) vec_error[c] = 0;

        // NEGATIVE SAMPLING
        for (int d = 0; d != num_negative + 1; d++)
        {
            if (d == 0)
            {
                target = w;
                label = 1;
            }
            else
            {
                do{
                    target = mw_neg_table[Rand(seed)];
                }while(target != w);
                label = 0;
            }
            lw = target * dim;
            Update(&emb_vertex[lm], &emb_vertex[lw], vec_error, label);
        }
        for (int c = 0; c != dim; c++) emb_vertex[c + lm] += vec_error[c];
		count++;
	}
	free(vec_error);
	pthread_exit(NULL);
}

void *TrainUWLINEThread(void *id)
{
	long long u, w, lu, lw, target, label;
	long long count = 0, last_count = 0, curedge;
	unsigned long long seed = (long long)id;
	real *vec_error = (real *)calloc(dim, sizeof(real));


	while (1)
	{
		//judge for exit
		if (count > uw_total_samples / num_threads + 2) break;

		if (count - last_count > 10000)
		{
			uw_current_sample_count += count - last_count;
			last_count = count;
			printf("%cRho: %f  train UW Progress: %.3lf%%", 13, rho, (real)uw_current_sample_count / (real)(uw_total_samples + 1) * 100);
			fflush(stdout);
//			rho = init_rho * (1 - current_sample_count / (real)(total_samples + 1));
//			if (rho < init_rho * 0.0001) rho = init_rho * 0.0001;
		}

        curedge = SampleUWAnEdge(gsl_rng_uniform(gsl_r), gsl_rng_uniform(gsl_r));
        u = uw_edge_source_id[curedge];
        w = uw_edge_target_id[curedge];

        lu = u * dim;
        for (int c = 0; c != dim; c++) vec_error[c] = 0;

        // NEGATIVE SAMPLING
        for (int d = 0; d != num_negative + 1; d++)
        {
            if (d == 0)
            {
                target = w;
                label = 1;
            }
            else
            {
                do{
                    target = uw_neg_table[Rand(seed)];
                }while(target != w);
                label = 0;
            }
            lw = target * dim;
            Update(&emb_context[lu], &emb_vertex[lw], vec_error, label);
        }
        for (int c = 0; c != dim; c++) emb_context[c + lu] += vec_error[c];
		count++;
	}
	free(vec_error);
	pthread_exit(NULL);
}

void *TrainUMLINEThread(void *id)
{
	long long u, m, lu, lm, target, label;
	long long count = 0, last_count = 0, curedge;
	unsigned long long seed = (long long)id;
	real *vec_error = (real *)calloc(dim, sizeof(real));


	while (1)
	{
		//judge for exit
		if (count > um_total_samples / num_threads + 2) break;

		if (count - last_count > 10000)
		{
			um_current_sample_count += count - last_count;
			last_count = count;
			printf("%cRho: %f  train UM Progress: %.3lf%%", 13, rho, (real)um_current_sample_count / (real)(um_total_samples + 1) * 100);
			fflush(stdout);
//			rho = init_rho * (1 - current_sample_count / (real)(total_samples + 1));
//			if (rho < init_rho * 0.0001) rho = init_rho * 0.0001;
		}

        curedge = SampleUMAnEdge(gsl_rng_uniform(gsl_r), gsl_rng_uniform(gsl_r));
        u = um_edge_source_id[curedge];
        m = um_edge_target_id[curedge];

        lu = u * dim;
        for (int c = 0; c != dim; c++) vec_error[c] = 0;

        // NEGATIVE SAMPLING
        for (int d = 0; d != num_negative + 1; d++)
        {
            if (d == 0)
            {
                target = m;
                label = 1;
            }
            else
            {
                do{
                    target = um_neg_table[Rand(seed)];
                }while(target != m);
                label = 0;
            }
            lm = target * dim;
            Update(&emb_vertex[lu], &emb_vertex[lm], vec_error, label);
        }
        for (int c = 0; c != dim; c++) emb_vertex[c + lu] += vec_error[c];
		count++;
	}
	free(vec_error);
	pthread_exit(NULL);
}



void Output_emb()
{
	FILE *fo = fopen(embedding_file, "wb");
	fprintf(fo, "%d %d\n", num_vertices, dim);
	for (int a = 0; a < num_vertices; a++)
	{
		fprintf(fo, "%s ", vertex[a].name);
		if (is_binary) for (int b = 0; b < dim; b++) fwrite(&emb_vertex[a * dim + b], sizeof(real), 1, fo);
		else for (int b = 0; b < dim; b++) fprintf(fo, "%lf ", emb_vertex[a * dim + b]);
		fprintf(fo, "\n");
	}
	fclose(fo);
}

void Output_ctx()
{
	FILE *fo = fopen(context_file, "wb");
	fprintf(fo, "%d %d\n", num_vertices, dim);
	for (int a = 0; a < num_vertices; a++)
	{
		fprintf(fo, "%s ", vertex[a].name);
		if (is_binary) for (int b = 0; b < dim; b++) fwrite(&emb_context[a * dim + b], sizeof(real), 1, fo);
		else for (int b = 0; b < dim; b++) fprintf(fo, "%lf ", emb_context[a * dim + b]);
		fprintf(fo, "\n");
	}
	fclose(fo);
}


void TrainHUELINE() {
	long a;

	printf("--------------------------------\n");
	printf("MW Samples: %lldM\n", mw_total_samples / 1000000);
	printf("UM Samples: %lldM\n", um_total_samples / 1000000);
	printf("UW Samples: %lldM\n", uw_total_samples / 1000000);
	printf("Negative: %d\n", num_negative);
	printf("Dimension: %d\n", dim);
	printf("Initial rho: %lf\n", init_rho);
	printf("Iteration: %d\n", iteration)
	printf("Thread %d\n", num_threads);
	printf("--------------------------------\n");

	InitHashTable();

	LoadVertex();
	LoadEdges();
	InitVector();

	// edge samplings
	InitMWAliasTable();
	InitUWAliasTable();
	InitUMAliasTable();

    //  negative sampling;
	InitMWNegTable();
	InitUWNegTable();
	InitUMNegTable();

	// fast simgoid computation
	InitSigmoidTable();

    // random seed;
	gsl_rng_env_setup();
	gsl_T = gsl_rng_rand48;
	gsl_r = gsl_rng_alloc(gsl_T);
	gsl_rng_set(gsl_r, 314159265);

	clock_t start = clock();
	pthread_t *pt;
	for(int i=0; i< iteration; i++){
	    mw_current_sample_count = 0;
	    uw_current_sample_count = 0;
	    um_current_sample_count = 0;
	    printf("iteration %d       \n", (i+1));
	    printf("--------------------------------\n");
	    printf("--------------- train mw network --------------------\n");
	    pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
	    for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainMWLINEThread, (void *)a);
	    for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
	    free(pt);
	    printf("--------------- train um network --------------------\n");
	    pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
	    for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainUMLINEThread, (void *)a);
	    for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
	    free(pt);
	    printf("--------------- train uw network --------------------\n");
	    pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
	    for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainUWLINEThread, (void *)a);
	    for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
	    printf("--------------------------------\n");
	    free(pt);
//	    rho = init_rho * (1 - i / (real)(iteration));
//		if (rho < init_rho * 0.0001) rho = init_rho * 0.0001;
	}
	printf("\n");
	clock_t finish = clock();
	printf("Total time: %lf\n", (double)(finish - start) / CLOCKS_PER_SEC);

	Output_emb();
	Output_ctx();
}

int ArgPos(char *str, int argc, char **argv) {
	int a;
	for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
		if (a == argc - 1) {
			printf("Argument missing for %s\n", str);
			exit(1);
		}
		return a;
	}
	return -1;
}

int main(int argc, char **argv) {
	int i;
	if (argc == 1) {
        printf("LINE: Large Information Network Embedding\n\n");
        printf("Options:\n");
        printf("Parameters for training:\n");
        printf("\t-trainuw <file>\n");
        printf("\t\tUse user word network data from <file> to train the model\n");
        printf("\t-trainww <file>\n");
        printf("\t\tUse word word network data from <file> to train the model\n");
        printf("\t-output <file>\n");
        printf("\t\tUse <file> to save the learnt embeddings\n");
        printf("\t-binary <int>\n");
        printf("\t\tSave the learnt embeddings in binary moded; default is 0 (off)\n");
        printf("\t-size <int>\n");
        printf("\t\tSet dimension of vertex embeddings; default is 100\n");
        printf("\t-order <int>\n");
        printf("\t\tThe type of the model; 1 for first order, 2 for second order; default is 2\n");
        printf("\t-negative <int>\n");
        printf("\t\tNumber of negative examples; default is 5\n");
        printf("\t-samples <int>\n");
        printf("\t\tSet the number of training samples as <int>Million; default is 1\n");
        printf("\t-threads <int>\n");
        printf("\t\tUse <int> threads (default 1)\n");
        printf("\t-rho <float>\n");
        printf("\t\tSet the starting learning rate; default is 0.025\n");
        printf("\nExamples:\n");
        printf("./line -trainuw uwnet.txt -trainww wwnet.txt -output vec.txt -binary 1 -size 200 -order 2 -negative 5 -samples 100 -rho 0.025 -threads 20\n\n");
        return 0;
	}
	if ((i = ArgPos((char *)"-trainmw", argc, argv)) > 0) strcpy(message_word_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-trainuw", argc, argv)) > 0) strcpy(user_word_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-trainum", argc, argv)) > 0) strcpy(user_message_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-outputemb", argc, argv)) > 0) strcpy(embedding_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-outputctx", argc, argv)) > 0) strcpy(context_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) is_binary = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-size", argc, argv)) > 0) dim = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) num_negative = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-mw_samples", argc, argv)) > 0) mw_total_samples = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-uw_samples", argc, argv)) > 0) uw_total_samples = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-um_samples", argc, argv)) > 0) um_total_samples = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-rho", argc, argv)) > 0) init_rho = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-iter", argc, argv)) > 0) iteration = atoi(argv([ i+1 ]))
	if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);

	mw_total_samples *= 1000000;
	uw_total_samples *= 1000000;
	um_total_samples *= 1000000;

	rho = init_rho;
	vertex = (struct ClassVertex *)calloc(max_num_vertices, sizeof(struct ClassVertex));
	TrainHUELINE();
	return 0;
}
