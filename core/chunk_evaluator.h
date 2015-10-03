#ifndef __CHUNK_EVALUATOR_H__
#define __CHUNK_EVALUATOR_H__

typedef char* char_ptr;
typedef void* void_ptr;
typedef size_t* size_ptr;

void init_random_seed();

/**
A string_buffer_t stores a sequence of strings using -
- strings: a '\0'-separated char array
- indices: an array of integers containing the start position in 'strings' (5 bytes) of each string and its length (3 bytes)
For instance 'hello\0how\0are\0you' is stored as:
- [h e l l o \0 h o w \0 a r e \0 y o u]
- [(0, 0) (6, 3) (10, 3) (14, 3)]
*/
typedef struct string_buffer {
    char_ptr strings;
    int char_count;   		        // number of chars in strings including '\0'
    size_ptr indices;
    int string_count;  		        // number of strings
} string_buffer_t;

typedef string_buffer_t* string_buffer_ptr;

string_buffer_ptr dz_string_buffer_create(int string_count, int char_count);
string_buffer_ptr dz_string_buffer_make(char_ptr strings, int char_count, size_ptr indices, int string_count);
string_buffer_ptr dz_string_buffer_make_random(int string_count);
char_ptr dz_string_buffer_to_string(string_buffer_ptr sb);

typedef enum expr_node_type {
    EXPR_NODE_TYPE_UNDEF = 0,
    EXPR_NODE_TYPE_STRING_CONST = 1,
    EXPR_NODE_TYPE_INT_CONST = 2,
    EXPR_NODE_TYPE_FLOAT_CONST = 3,
	EXPR_NODE_TYPE_STRING_BUFFER = 4,
    EXPR_NODE_TYPE_STRING_BUFFER_INDEX = 5,
    EXPR_NODE_TYPE_UPPER = 6,
    EXPR_NODE_TYPE_LOWER = 7,
    EXPR_NODE_TYPE_CONCAT = 8
} expr_node_type_t;

typedef struct expr_node expr_node_t;
typedef expr_node_t* expr_node_ptr;

struct expr_node {
  expr_node_type_t type;
  long long int_arg;
  double float_arg;
  char_ptr string_arg;
  string_buffer_ptr string_buffer_arg;
  int string_buffer_index_arg;
  expr_node_ptr args[3];
};


expr_node_ptr dz_expr_node_int_create(long long x);
expr_node_ptr dz_expr_node_float_create(double x);
expr_node_ptr dz_expr_node_string_create(char_ptr s);
expr_node_ptr dz_expr_node_string_buffer_index_create(int sb_index);
expr_node_ptr dz_expr_node_create(expr_node_type_t type, void_ptr arg1, void_ptr arg2, void_ptr arg3);

long long dz_expr_node_get_int(expr_node_ptr n);
double dz_expr_node_get_float(expr_node_ptr n);
char_ptr dz_expr_node_get_string(expr_node_ptr n);
int dz_expr_node_get_string_buffer_index(expr_node_ptr n);
expr_node_ptr dz_expr_node_get_arg(expr_node_ptr n, int i);

expr_node_ptr dz_eval(expr_node_ptr n, string_buffer_ptr string_buffers[]);

#endif // __CHUNK_EVALUATOR_H__