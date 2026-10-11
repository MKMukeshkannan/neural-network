#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define ARENA_IMPLEMENTATION
#include "./arena.h"

#ifdef DEBUG
#define dprintf(...) printf(...)
#else
#define dprintf(...) 0
#endif

typedef int8_t   s8;
typedef int16_t  s16;
typedef int32_t  s32;
typedef int64_t  s64;
typedef uint8_t  u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
typedef float    f32;
typedef double   f64;
#define __internal static

#define KB(x) ((size_t)(x) * 1024ULL)
#define MB(x) (KB(x) * 1024ULL)
#define GB(x) (MB(x) * 1024ULL)

const size_t NUM_OF_PARAMS = 7;
const size_t NUM_OF_SAMPLES = 1000;
const size_t ARENA_SIZE = KB(100);
const f64    LEARNING_RATE = 0.01f;
const f64 mean = 618861.0186467685;
const f64 std = 253568.05837542083;

Arena g_arena;


/* HEADER MESS */
typedef struct { f64* data; size_t row; size_t clm; } Matrix;
void mat_create(Matrix* mat, size_t row, size_t clm) { assert(mat); mat->data = arena_alloc_purpose(&g_arena, row * clm * sizeof*mat->data, "matrix_creation"); mat->row = row; mat->clm = clm; memset(mat->data, 0, row*clm*sizeof(f64)); }
f64 mat_get(Matrix* mat, size_t row, size_t clm) { assert(mat); assert(mat->row > row && mat->clm > clm); return mat->data[row * mat->clm + clm]; }
f64* mat_get_ref(Matrix* mat, size_t row, size_t clm) { assert(mat); assert(mat->row > row && mat->clm > clm); return &mat->data[row * mat->clm + clm]; }
void mat_copy(Matrix* mat, f64 values[], size_t size) { assert(mat); for (size_t i = 0; i < size; ++i) mat->data[i] = values[i]; }
void mat_print(Matrix*mat) { for (size_t i = 0; i < mat->row; ++i) { for (size_t j = 0; j < mat->clm; ++j) dprintf("%f ", mat_get(mat, i, j)); dprintf("\n\t"); } }
void mat_vmul(Matrix* a, Matrix* b, Matrix* res) { assert(a->clm == b->row); assert(res->row == a->row); assert(res->clm == b->clm); memset(res->data, 0, res->row*res->clm*sizeof*res->data); for (size_t i = 0; i < a->row; ++i) for (size_t j = 0; j < b->clm; ++j) for (size_t k = 0; k < a->clm; ++k) *mat_get_ref(res, i, j) += mat_get(a, i, k) * mat_get(b, k, j); }
void mat_transpose_into(Matrix* src, Matrix* dst) { for (size_t i = 0; i < src->row; ++i) for (size_t j = 0; j < src->clm; ++j) *mat_get_ref(dst, j, i) = mat_get(src, i, j); }
void mat_vadd(Matrix* a, Matrix* b, Matrix* res) { assert(a->clm == b->clm && a->row == b->row && res->row == a->row && res->clm == a->clm); for (size_t i = 0; i < a->row; ++i) for (size_t j = 0; j < a->clm; ++j) *mat_get_ref(res, i, j) = mat_get(a, i, j) + mat_get(b, i, j); }

typedef struct { f64* data; size_t len; size_t cap; } Array;
void array_create(Array* ar, size_t len) { ar->data = arena_alloc_purpose(&g_arena, sizeof*ar->data * len, "array_creation"); ar->len = 0; ar->cap = len; }
void array_push(Array* ar, f64 data) { assert(ar->len<ar->cap); ar->data[ar->len++] = data; }
f64  array_get_data(Array* ar, size_t row, size_t param) { assert(ar && ar->data && ar->len > 0); assert(row < NUM_OF_SAMPLES); assert(param < NUM_OF_PARAMS); return ar->data[row * NUM_OF_PARAMS + param]; }
f64* array_get_row(Array* ar, size_t row) { assert(ar); assert(row<ar->len); return &ar->data[row * NUM_OF_PARAMS]; }

typedef struct { Matrix* data; size_t len; size_t cap; } ArrMat;
void arrmat_create(ArrMat* ar, size_t len) { ar->data = arena_alloc_purpose(&g_arena, sizeof*ar->data * len, "arrmat_creation"); ar->len = 0; ar->cap = len; memset(ar->data, 0, sizeof*ar->data*len); }
void arrmat_push(ArrMat* ar, Matrix data) { assert(ar->len<ar->cap); ar->data[ar->len++] = data; }
Matrix* arrmat_get(const ArrMat* arr_mat, size_t i) { assert(arr_mat); assert(i < arr_mat->len); return &arr_mat->data[i]; }

// HELPERS
__internal void read_csv(Array* X, Array* Y);
f64 relu(f64 x) { return fmax(x, 0); }
f64 relu_prime(f64 x) {    return x > 0.0 ? 1.0 : 0.0; }

typedef struct 
{
    ArrMat weights;
    ArrMat bias;
    ArrMat m_neuron_layers; // z = w.x + b
    ArrMat m_neuron_loss_layers; // 
    size_t number_of_inputs;
    size_t number_of_outputs;
} NeuralNetwork;
NeuralNetwork* nn_create(size_t number_of_neurons_in_each_layer[], size_t number_of_layers);
bool nn_save_text(const NeuralNetwork* nn, const char* filename);
bool nn_load_text(NeuralNetwork* nn, const char* filename);
void nn_print_weights(const NeuralNetwork* nn);

void nn_forward_feed(NeuralNetwork* nn, f64 inputs[], size_t len);
void nn_backprop(NeuralNetwork* nn, f64* Y, size_t size);
f64 predict(NeuralNetwork* nn, f64 inputs[], size_t len) { dprintf(":: PREDICTION PIPELIN BEGIN ::\n"); nn_forward_feed(nn, inputs, len); dprintf(":: END PREDICTION PIPELIN  ::\n"); return nn->m_neuron_layers.data[nn->m_neuron_layers.len - 1].data[0]; }
void nn_print_weights_and_bias (NeuralNetwork* nn) { assert(nn); dprintf("\nPHASE: PRINTING WEIGHTS AND BIAS\n\t"); for (size_t i = 0; i < nn->weights.len; ++i) { Matrix* weight = arrmat_get(&nn->weights, i); Matrix* bias = arrmat_get(&nn->bias, i); dprintf("\"weight of %zu\"\n\t", i); mat_print(weight); dprintf("\"bias of %zu\"\n\t", i); mat_print(bias); } printf("\n"); }
void nn_print_each_neurons(NeuralNetwork *nn) { assert(nn); dprintf("\nPHASE: PRINTING EACH NEURON     \n"); for (size_t i = 0; i < nn->m_neuron_layers.len; ++i) { printf("\t"); Matrix* neuron = arrmat_get(&nn->m_neuron_layers, i); mat_print(neuron); dprintf("\n"); } dprintf("\n"); }

f64 random_normal(f64 mean, f64 stddev); f64 mrand(void) {return random_normal(0, 0.1);}
void randomize_weights_and_bias(NeuralNetwork* nn) { for (size_t i = 0; i < nn->weights.len; ++i) { Matrix* mat = arrmat_get(&nn->weights, i); f64 fan_in = (f64)mat->clm; f64 stddev = sqrt(2.0 / fan_in); for (size_t j = 0; j < mat->row * mat->clm; ++j) { mat->data[j] = random_normal(0.0, stddev); } } for (size_t i = 0; i < nn->bias.len; ++i) { Matrix* mat = arrmat_get(&nn->bias, i); for (size_t j = 0; j < mat->row * mat->clm; ++j) { mat->data[j] = 0.0; } } }

// NOTE: works only for single output
f64 get_mse(NeuralNetwork *nn, Array *X, Array *Y) { f64 res = 0; for (size_t i = 0; i < NUM_OF_SAMPLES; ++i) { f64 *rw = array_get_row(X, i); f64 temp = (predict(nn, rw, NUM_OF_PARAMS) - Y->data[i]); res += temp * temp; } res = (1.0f / (2 * NUM_OF_SAMPLES)) * res; return res; }
void nn_clear_weights_and_bias(NeuralNetwork* nn) { assert(nn); for (size_t ilayer = 0; ilayer < nn->m_neuron_loss_layers.len; ++ilayer) { Matrix* mat = arrmat_get(&nn->m_neuron_loss_layers, ilayer); memset(mat->data, 0, mat->row * mat->clm * sizeof(f64)); } for (size_t ilayer = 0; ilayer < nn->m_neuron_layers.len; ++ilayer) { Matrix* mat = arrmat_get(&nn->m_neuron_layers, ilayer); memset(mat->data, 0, mat->row * mat->clm * sizeof(f64)); } }


void nn_train(NeuralNetwork* nn, Array* X, Array* Y)
{
    for (size_t epoch = 0; epoch < 1000; ++epoch)
    {
        f64 mse = get_mse(nn, X, Y);
        printf("epoch: %zu %.17lf\n", epoch, mse);

        for (size_t ix = 0; ix < NUM_OF_SAMPLES; ++ix)
        {
            predict(nn, array_get_row(X, ix), NUM_OF_PARAMS);
            nn_backprop(nn, &Y->data[ix], 1);
            nn_clear_weights_and_bias(nn);
        }
    }
}


int main(void)
{
    srand(time(NULL));

    u8* buf = malloc(ARENA_SIZE);
    arena_init(&g_arena, buf, ARENA_SIZE);

    Array X, Y;
    array_create(&X, NUM_OF_PARAMS * NUM_OF_SAMPLES);
    array_create(&Y, 1 * NUM_OF_SAMPLES);

    read_csv(&X, &Y);

    for (size_t i = 0; i < NUM_OF_SAMPLES; ++i) dprintf("%f %f\n", *array_get_row(&X, i), Y.data[i]);

    NeuralNetwork* nn = nn_create((size_t[]){7,16,8,1}, 4);
    randomize_weights_and_bias(nn);

    dprintf(":: INITIAL WEIGHTS and BISAS");
    nn_print_weights_and_bias(nn);

    nn_train(nn, &X, &Y);
    nn_save_text(nn, "model");

    arena_destroy(&g_arena);
    return 0;
}


// @helper.readcsv                                                                                          .
__internal void read_csv(Array* X, Array* Y)
{

    dprintf("PHASE: READING CSV \n");
    FILE* file = fopen("./data/train.csv", "r");
    if (!file)
    {
        dprintf("unable to read file!!\n");
        return;
    }

    (void) fseek(file, 0, SEEK_END);
    size_t total_size = ftell(file);
    dprintf("\ttotal_file_size: %zu\n", total_size);
    (void) fseek(file, 0L, SEEK_SET);

    u8* bytes = (u8*)malloc(total_size + 1);
    if (bytes == NULL)
        return;

    fread(bytes, sizeof(*bytes), total_size + 1, file);
    bytes[total_size] = '\0';
    dprintf("\tbytes_read: %zu\n", read);


    u8* p = bytes;
    while (*p != '\n') p++; // SKIP THE HEADER 
    p++;

    char buf[100]; size_t ibuf = 0;
    while (*p != '\0')
    {
        if (Y->len == Y->cap)
        {
            free(bytes);
            return;
        }

        if (*p == ',')
        {
            f64 res = strtod(buf, NULL);
            array_push(X, res);
            memset(buf, 0, 100);
            ibuf = 0;
        }
        else if (*p == '\n')
        {
            f64 res = strtod(buf, NULL);
            array_push(Y, res);
            memset(buf, 0, 100);
            ibuf = 0;
        }
        else 
        {
            if (ibuf < 100)
                buf[ibuf++] = *p;
        }

        p++;
    }
    free(bytes);
    dprintf("\tsucessfully imported data\n");
}

f64 random_normal(f64 mean, f64 stddev)
{
    f64 u1, u2;

    // Uniform random numbers in (0, 1)
    u1 = (rand() + 1.0) / (RAND_MAX + 2.0);
    u2 = (rand() + 1.0) / (RAND_MAX + 2.0);

    // Box-Muller transform
    f64 z = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);

    return mean + stddev * z;
}

NeuralNetwork* nn_create(size_t number_of_neurons_in_each_layer[], size_t number_of_layers)
{
    NeuralNetwork* res = (NeuralNetwork*)arena_alloc_purpose(&g_arena, sizeof(NeuralNetwork), "neural_network");
    res->number_of_inputs = number_of_neurons_in_each_layer[0];
    res->number_of_outputs = number_of_neurons_in_each_layer[number_of_layers-1];

    arrmat_create(&res->weights, number_of_layers - 1);
    arrmat_create(&res->bias, number_of_layers - 1);
    arrmat_create(&res->m_neuron_layers, number_of_layers);
    arrmat_create(&res->m_neuron_loss_layers, number_of_layers);

    dprintf("\nPHASE: CREATING NEURAL NETWORK\n");

    for (size_t cur_layer = 0; cur_layer < number_of_layers; ++cur_layer)
    {
        size_t prev_no_nodes = number_of_neurons_in_each_layer[cur_layer], 
               curr_no_nodes = cur_layer == number_of_layers - 1 ? 0 : number_of_neurons_in_each_layer[cur_layer + 1];

        if (cur_layer < number_of_layers - 1)
        {
            Matrix wei_mat; mat_create(&wei_mat, curr_no_nodes, prev_no_nodes);
            arrmat_push(&res->weights, wei_mat);
            memset(wei_mat.data, 0, wei_mat.clm * wei_mat.row * sizeof*wei_mat.data);
            dprintf("\tweight_matrix b/w (%zu-%zu): %zu, %zu\n", cur_layer, cur_layer+1, wei_mat.row, wei_mat.clm);

            Matrix bias_vec; mat_create(&bias_vec, curr_no_nodes, 1L);
            arrmat_push(&res->bias, bias_vec);
            memset(bias_vec.data, 0, bias_vec.clm * bias_vec.row * sizeof*bias_vec.data);
            dprintf("\tbias_matrix b/w (%zu-%zu): %zu, %zu\n", cur_layer, cur_layer+1, bias_vec.row, bias_vec.clm);
        }

        Matrix neuron_vec; mat_create(&neuron_vec, prev_no_nodes, 1L);
        arrmat_push(&res->m_neuron_layers, neuron_vec);
        memset(neuron_vec.data, 0, neuron_vec.clm * neuron_vec.row * sizeof*neuron_vec.data);
        dprintf("\tneuron_matrix b/w (%zu-%zu): %zu, %zu\n", cur_layer, cur_layer+1, neuron_vec.row, neuron_vec.clm);

        Matrix neuron_loss_vec; mat_create(&neuron_loss_vec, prev_no_nodes, 1L);
        arrmat_push(&res->m_neuron_loss_layers, neuron_loss_vec);
        memset(neuron_loss_vec.data, 0, neuron_loss_vec.clm * neuron_loss_vec.row * sizeof*neuron_loss_vec.data);
        dprintf("\tneuron_loss_vector b/w (%zu-%zu): %zu, %zu\n\n", cur_layer, cur_layer+1, neuron_vec.row, neuron_vec.clm);
    }

    return res;
}

bool nn_save_text(const NeuralNetwork* nn, const char* filename) {
    FILE* fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Error: Could not open file '%s' for writing: \n", filename);
        return false;
    }
    
    size_t num_layers = nn->weights.len + 1;
    fprintf(fp, "# Neural Network Weights and Biases\n");
    fprintf(fp, "# Format: layer_sizes, then weight matrices, then bias vectors\n");
    fprintf(fp, "layers %zu\n", num_layers);
    
    fprintf(fp, "layer_sizes");
    for (size_t i = 0; i < num_layers; i++) {
        fprintf(fp, " %zu", nn->m_neuron_layers.data[i].row);
    }
    fprintf(fp, "\n");
    
    fprintf(fp, "\n# Weight matrices (rows = neurons in next layer, cols = neurons in current layer)\n");
    for (size_t i = 0; i < nn->weights.len; i++) {
        const Matrix* w = arrmat_get(&nn->weights, i);
        fprintf(fp, "\nweight_layer %zu %zu %zu\n", i, w->row, w->clm);
        for (size_t r = 0; r < w->row; r++) {
            for (size_t c = 0; c < w->clm; c++) {
                fprintf(fp, "%.17f ", mat_get((Matrix*)w, r, c));
            }
            fprintf(fp, "\n");
        }
    }
    
    fprintf(fp, "\n# Bias vectors\n");
    for (size_t i = 0; i < nn->bias.len; i++) {
        const Matrix* b = arrmat_get(&nn->bias, i);
        fprintf(fp, "\nbias_layer %zu %zu %zu\n", i, b->row, b->clm);
        for (size_t r = 0; r < b->row; r++) {
            fprintf(fp, "%.17f\n", mat_get((Matrix*)b, r, 0));
        }
    }
    
    fprintf(fp, "\n# End of file\n");
    fclose(fp);
    return true;
}

bool nn_load_text(NeuralNetwork* nn, const char* filename) {
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Error: Could not open file '%s' for reading: \n", filename);
        return false;
    }
    
    char buffer[256];
    size_t num_layers = 0;
    size_t* layer_sizes = NULL;
    
    while (fgets(buffer, sizeof(buffer), fp)) {
        if (buffer[0] == '#' || buffer[0] == '\n') continue;
        
        if (strncmp(buffer, "layers", 6) == 0) {
            if (sscanf(buffer, "layers %zu", &num_layers) != 1) {
                fprintf(stderr, "Error: Invalid layers format\n");
                fclose(fp);
                return false;
            }
            layer_sizes = (size_t*)malloc(num_layers * sizeof(size_t));
            if (!layer_sizes) {
                fclose(fp);
                return false;
            }
        }
        
        if (strncmp(buffer, "layer_sizes", 11) == 0) {
            if (!layer_sizes) {
                fprintf(stderr, "Error: layer_sizes before layers definition\n");
                fclose(fp);
                return false;
            }
            char* token = strtok(buffer, " ");
            token = strtok(NULL, " "); // Skip "layer_sizes"
            for (size_t i = 0; i < num_layers && token != NULL; i++) {
                layer_sizes[i] = atoi(token);
                token = strtok(NULL, " ");
            }
        }
        
        if (strncmp(buffer, "weight_layer", 12) == 0) {
            size_t layer_idx, rows, cols;
            if (sscanf(buffer, "weight_layer %zu %zu %zu", &layer_idx, &rows, &cols) != 3) {
                fprintf(stderr, "Error: Invalid weight_layer format\n");
                free(layer_sizes);
                fclose(fp);
                return false;
            }
            
            Matrix* w = arrmat_get(&nn->weights, layer_idx);
            if (!w) {
                fprintf(stderr, "Error: Weight matrix %zu not initialized\n", layer_idx);
                free(layer_sizes);
                fclose(fp);
                return false;
            }
            
            for (size_t r = 0; r < rows; r++) {
                char line[1024];
                if (!fgets(line, sizeof(line), fp)) {
                    fprintf(stderr, "Error: Unexpected end of file while reading weights\n");
                    free(layer_sizes);
                    fclose(fp);
                    return false;
                }
                
                char* token = strtok(line, " \t\n");
                for (size_t c = 0; c < cols && token != NULL; c++) {
                    *mat_get_ref(w, r, c) = atof(token);
                    token = strtok(NULL, " \t\n");
                }
            }
        }
        
        if (strncmp(buffer, "bias_layer", 10) == 0) {
            size_t layer_idx, rows, cols;
            if (sscanf(buffer, "bias_layer %zu %zu %zu", &layer_idx, &rows, &cols) != 3) {
                fprintf(stderr, "Error: Invalid bias_layer format\n");
                free(layer_sizes);
                fclose(fp);
                return false;
            }
            
            Matrix* b = arrmat_get(&nn->bias, layer_idx);
            if (!b) {
                fprintf(stderr, "Error: Bias vector %zu not initialized\n", layer_idx);
                free(layer_sizes);
                fclose(fp);
                return false;
            }
            
            for (size_t r = 0; r < rows; r++) {
                char line[256];
                if (!fgets(line, sizeof(line), fp)) {
                    fprintf(stderr, "Error: Unexpected end of file while reading biases\n");
                    free(layer_sizes);
                    fclose(fp);
                    return false;
                }
                *mat_get_ref(b, r, 0) = atof(line);
            }
        }
    }
    
    free(layer_sizes);
    fclose(fp);
    return true;
}

void nn_print_weights(const NeuralNetwork* nn) 
{
    printf("\n=== Neural Network Weights ===\n");
    printf("Layers: %zu\n", nn->weights.len + 1);
    
    for (size_t i = 0; i < nn->weights.len; i++) 
    {
        const Matrix* w = arrmat_get(&nn->weights, i);
        printf("\nWeight Layer %zu (%zux%zu):\n", i, w->row, w->clm);
        mat_print((Matrix*)w);
    }
    
    printf("\n=== Biases ===\n");
    for (size_t i = 0; i < nn->bias.len; i++) 
    {
        const Matrix* b = arrmat_get(&nn->bias, i);
        printf("\nBias Layer %zu (%zux%zu):\n", i, b->row, b->clm);
        mat_print((Matrix*)b);
    }
}

void nn_forward_feed(NeuralNetwork* nn, f64 inputs[], size_t len)
{
    TempArena temp_arena = temp_arena_begin(&g_arena);

    dprintf("\t> got inputs: ");
    for (size_t i = 0; i < len; ++i) dprintf("%f ", inputs[i]);
    dprintf("\n");

    Matrix ip_mat; 
    mat_create(&ip_mat, len, 1);
    mat_copy(&ip_mat, inputs, len);
    mat_copy(&nn->m_neuron_layers.data[0], inputs, 1L*len);

    Matrix res_mat; 
    mat_create(&res_mat, arrmat_get(&nn->weights, 0)->row, 1);

    for (size_t ilayer = 0; ilayer < nn->m_neuron_layers.len - 1; ++ilayer) 
    {
        mat_vmul(arrmat_get(&nn->weights, ilayer), &ip_mat, &res_mat);
        mat_vadd(arrmat_get(&nn->bias, ilayer), &res_mat, &res_mat);

        mat_copy(&nn->m_neuron_layers.data[ilayer + 1], res_mat.data, res_mat.clm*res_mat.row);

        dprintf("\t> unact :");
        for (size_t i = 0; i < res_mat.row*res_mat.clm; ++i) dprintf("%.17f ", res_mat.data[i]);
        dprintf("\n");

        for (size_t i = 0; i < res_mat.row*res_mat.clm; ++i)
            res_mat.data[i] = relu(res_mat.data[i]);

        dprintf("\t> act :");
        for (size_t i = 0; i < res_mat.row*res_mat.clm; ++i) dprintf("%f ", res_mat.data[i]);
        dprintf("\n");

        // result becomes new input for next layer
        mat_create(&ip_mat, res_mat.row, res_mat.clm);
        mat_copy(&ip_mat, res_mat.data, res_mat.row*res_mat.clm);

        if (ilayer < nn->m_neuron_layers.len - 2)
            mat_create(&res_mat, arrmat_get(&nn->weights, ilayer + 1)->row, res_mat.clm);
    }

    temp_arena_end(&temp_arena);
}

void nn_backprop(NeuralNetwork* nn, f64* Y, size_t size)
{
    TempArena temp_arena = temp_arena_begin(&g_arena);

    size_t total_layer = nn->m_neuron_layers.len;

    // LOSS FOR LAST LAYER, contains unactivated_vector: W x I + B
    {

        Matrix loss_vector = nn->m_neuron_loss_layers.data[total_layer - 1];
        Matrix unactivated_vector = nn->m_neuron_layers.data[total_layer - 1];
        Matrix y_vector; mat_create(&y_vector, size, 1L); mat_copy(&y_vector, Y, size);

        // LOSS = (act(z) - res) * act'(z)
        assert(loss_vector.row == y_vector.row && 
               loss_vector.clm == y_vector.clm && 
               y_vector.row == unactivated_vector.row && 
               y_vector.clm == unactivated_vector.clm);

        for (size_t i = 0; i < loss_vector.row; ++i)
        {
            for (size_t j = 0; j < loss_vector.clm; ++j)
            {
                f64 prediction = mat_get(&unactivated_vector, i, j);
                f64 target = mat_get(&y_vector, i, j);

                *mat_get_ref(&loss_vector, i, j) = prediction - target;
            }
        }

    }
    
    // LOSS FOR REST OF THE LAYERS
    {
        for (int ilayer = total_layer - 2; ilayer >= 0; ilayer--)
        {
            Matrix* iwei = arrmat_get(&nn->weights, ilayer);
            Matrix* ip1loss_layer = arrmat_get(&nn->m_neuron_loss_layers, ilayer + 1);
            Matrix* iloss_layer = arrmat_get(&nn->m_neuron_loss_layers, ilayer);
            Matrix* iunactivated_vector = arrmat_get(&nn->m_neuron_layers, ilayer);

            Matrix wei_transpose; 
            mat_create(&wei_transpose, iwei->clm, iwei->row);
            mat_transpose_into(iwei, &wei_transpose);

            mat_vmul(&wei_transpose, ip1loss_layer, iloss_layer);

            for (size_t i = 0; i < iloss_layer->clm*iloss_layer->row; ++i)
                iloss_layer->data[i] *= relu_prime(iunactivated_vector->data[i]);
        }
    }

    dprintf(":: BACKPROP PIPELINE BEGIN %f ::", *Y);
    for (size_t  i = 0; i < nn->m_neuron_layers.len; ++i)
    {
        dprintf("\n\t> laye: %zu ", i);
        Matrix layer = nn->m_neuron_layers.data[i];
        for (size_t i = 0; i < layer.clm*layer.row; ++i)
            dprintf("%.17f ", layer.data[i]);

        dprintf("\n\t> loss: %zu ", i);
        Matrix loss = nn->m_neuron_loss_layers.data[i];
        for (size_t i = 0; i < loss.clm*loss.row; ++i)
            dprintf("%.17f ", loss.data[i]);
    }
    dprintf("\n:: END BACKPROP PIPELINE BEGIN ::\n");

    // Adjusting BIAS....................
    for (size_t i = 0; i < total_layer - 1; ++i)
    { 
        Matrix* bias = arrmat_get(&nn->bias, i);
        Matrix* loss = arrmat_get(&nn->m_neuron_loss_layers, i + 1);

        dprintf("BIAS\n");
        for (size_t i = 0; i < bias->row * bias->clm; ++i)
            dprintf("%.17f", bias->data[i]);

        dprintf("\n");
        dprintf("LOSS\n");
        for (size_t i = 0; i < loss->row * loss->clm; ++i)
            dprintf("%.17f", loss->data[i]);
        dprintf("\n");

        assert(bias->row == loss->row && bias->clm == loss->clm);

        for (size_t i = 0; i < bias->row * bias->clm; ++i)
            bias->data[i] -= LEARNING_RATE*loss->data[i];

        dprintf("ADJ BIAS\n");
        for (size_t i = 0; i < loss->row * loss->clm; ++i)
            dprintf("%.17f", bias->data[i]);
        dprintf("\n");
    }


    // Adjusting WEIGHT....................
    for (size_t ilayer = 0; ilayer < total_layer - 1; ++ilayer)
    {
        Matrix* weight = arrmat_get(&nn->weights, ilayer);

        Matrix* loss = arrmat_get(&nn->m_neuron_loss_layers, ilayer + 1);
        Matrix* tmp = arrmat_get(&nn->m_neuron_layers, ilayer);
        Matrix neuron; mat_create(&neuron, tmp->row, tmp->clm); mat_copy(&neuron, tmp->data, tmp->row*tmp->clm);

        if (ilayer != 0)
        {
            for (size_t i = 0; i < neuron.row*neuron.clm; ++i)
                neuron.data[i] = relu(neuron.data[i]);
        }

        Matrix weight_gradient; mat_create(&weight_gradient, loss->row, neuron.row);

        Matrix neuron_transpose;
        mat_create(&neuron_transpose, neuron.clm, neuron.row);
        mat_transpose_into(&neuron, &neuron_transpose);

        mat_vmul(loss, &neuron_transpose, &weight_gradient);

        for (size_t i = 0; i < weight_gradient.row*weight_gradient.clm; ++i)
            weight->data[i] -= LEARNING_RATE*weight_gradient.data[i];
    }

    temp_arena_end(&temp_arena);
}
