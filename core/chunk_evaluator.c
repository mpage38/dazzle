#include <Time.h>
#include <windows.h>

#include "chunk_evaluator.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>

/**
 * initialize random number seed
 */
void init_random_seed() {
	srand((int)time(NULL));  
}

/**
 Random integer generator.
 Assumes srand() already executed
*/
int dz_random_number(int min_num, int max_num) {
    int result = 0, low_num = 0, hi_num = 0;
    if ( min_num < max_num ) {
        low_num = min_num;
        hi_num = max_num + 1;
    }
    else {
        low_num = max_num + 1;
        hi_num = min_num;
    }
    result = (rand() % (hi_num - low_num)) + low_num;
    return result;
}

// -----------------------------------------------------------------

/**
 * Create expression node
 */
expr_node_ptr dz_expr_node_int_create(long long x) {
    expr_node_ptr node = (expr_node_ptr) malloc(sizeof(expr_node_t));
    node->type = EXPR_NODE_TYPE_INT_CONST;
	node->int_arg = x; 
    return node;
}

expr_node_ptr dz_expr_node_float_create(double x) {
    expr_node_ptr node = (expr_node_ptr) malloc(sizeof(expr_node_t));
    node->type = EXPR_NODE_TYPE_FLOAT_CONST;
	node->float_arg = x; 
    return node;
}

expr_node_ptr dz_expr_node_string_create(char_ptr s) {
    expr_node_ptr node = (expr_node_ptr) malloc(sizeof(expr_node_t));
	node->type = EXPR_NODE_TYPE_STRING_CONST;
	node->string_arg = s; 
    return node;
}

expr_node_ptr dz_expr_node_string_buffer_index_create(int sb_index) {
    expr_node_ptr node = (expr_node_ptr) malloc(sizeof(expr_node_t));
	node->type = EXPR_NODE_TYPE_STRING_BUFFER_INDEX;
	node->string_buffer_index_arg = sb_index; 
    return node;
}

expr_node_ptr dz_expr_node_string_buffer_create(string_buffer_ptr sb) {
    expr_node_ptr node = (expr_node_ptr) malloc(sizeof(expr_node_t));
	node->type = EXPR_NODE_TYPE_STRING_BUFFER;
	node->string_buffer_arg = sb; 
    return node;
}

expr_node_ptr dz_expr_node_create(expr_node_type_t type, void_ptr arg1, void_ptr arg2, void_ptr arg3) {
    expr_node_ptr node = (expr_node_ptr) malloc(sizeof(expr_node_t));
    node->type = type;
    node->args[0] = arg1;
    node->args[1] = arg2;
    node->args[2] = arg3;

    return node;
}


long long dz_expr_node_get_int(expr_node_ptr n) {
	long long x;
	
	assert(n->type == EXPR_NODE_TYPE_INT_CONST);
	x = n->int_arg;
	return x;
}

double dz_expr_node_get_float(expr_node_ptr n) {
	double x;
	
	assert(n->type == EXPR_NODE_TYPE_FLOAT_CONST);
	x = n->float_arg;
	return x;
}

char_ptr dz_expr_node_get_string(expr_node_ptr n) {
	char_ptr s;
	
	assert(n->type == EXPR_NODE_TYPE_STRING_CONST);
	s = n->string_arg;
	return s;
}

string_buffer_ptr dz_expr_node_get_string_buffer(expr_node_ptr n) {
	string_buffer_ptr sb;
	
	assert(n->type == EXPR_NODE_TYPE_STRING_BUFFER);
	sb = n->string_buffer_arg;
	return sb;
}

int dz_expr_node_get_string_buffer_index(expr_node_ptr n) {
	int sb_index;
	
	assert(n->type == EXPR_NODE_TYPE_STRING_BUFFER_INDEX);
	sb_index = n->string_buffer_index_arg;
	return sb_index;
}

expr_node_ptr dz_expr_node_get_arg(expr_node_ptr n, int i) {
	return (expr_node_ptr)n->args[i];
}

// -----------------------------------------------------------------

/*
 Print string_buffer_t sb
 */
void dz_string_buffer_print(string_buffer_ptr sb) {
    int i, j;
    char c;

    printf("\n");
    for (i = 0; i < 200; i++) {
        c = (char)(sb->strings[i]);
		if (c == 0) {
			printf(" , ");
			if (i > 100) {
				break;
			}
		}
        else {
			printf("%c", c);
		}
    }
    printf(" ... ");
	for (i = sb->char_count - 100; i > 0; i--) {
		if (sb->strings[i] == 0) {
			break;
		}
	}
	for (j = i; j < sb->char_count; j++) {
        c = (char)(sb->strings[j]);
		if  (c == 0 && j < sb->char_count - 1) {
			printf(" , ");
		}
		else if (c != 0) {
			printf("%c", c);
		}
    }
    printf("\n");
    for (i = 0; i < 20; i++) {
        printf("(%d, %d)", (long)((sb->indices[i]) >> 24), (long)((sb->indices[i]) & 0x0000000000FFFFFF));
    }
    printf(" ... ");
    for (i = sb->string_count - 20; i < sb->string_count; i++) {
        printf("(%d, %d)", (long)((sb->indices[i]) >> 24), (long)((sb->indices[i]) & 0x0000000000FFFFFF));
    }
    printf("\n");
}

/**
 * Create and return a string_buffer_t
 */
string_buffer_ptr dz_string_buffer_create(int string_count, int char_count) {
    string_buffer_ptr sb;

    sb = (string_buffer_ptr)malloc(sizeof(string_buffer_t));
    sb->strings = (char_ptr)malloc(char_count);
    sb->char_count = char_count;
    sb->string_count = string_count;
    sb->indices = (size_ptr)malloc(string_count * sizeof(size_t));
    return sb;
}

string_buffer_ptr dz_string_buffer_make(char_ptr strings, int char_count, size_ptr indices, int string_count) {
    string_buffer_ptr sb;

    sb = (string_buffer_ptr)malloc(sizeof(string_buffer_t));
    sb->strings = strings;
    sb->char_count = char_count;
	sb->indices = indices;
    sb->string_count = string_count;

    printf("in dz_string_buffer_make");
    dz_string_buffer_print(sb);

    return sb;
}

char_ptr dz_string_buffer_to_string(string_buffer_ptr sb) {
	return sb->strings;
}

/*
 *Generate and return a string_buffer_t. filled with random strings
 */
string_buffer_ptr dz_string_buffer_make_random(int string_count) {
    char_ptr tmp_strings;
    size_ptr indices;
    int i, l;
    char letter;
    int word_length, total_length;
    string_buffer_ptr sb;

    tmp_strings = (char_ptr)malloc(string_count * 10);
    indices = (size_ptr)malloc(string_count * sizeof(size_t));
    total_length = 0;
    for (i = 0; i < string_count; i++) {
        letter = (char)dz_random_number(97, 122);
        word_length = dz_random_number(2, 10);
        for (l = 0; l < word_length; l++) {
            tmp_strings[total_length + l] = letter;
        }
        tmp_strings[total_length + word_length] = 0;
        indices[i] = ((size_t)total_length << (size_t)24) + (size_t)word_length;
        total_length += (word_length + 1);
    }
    sb = dz_string_buffer_create(string_count, total_length);
    memcpy(&(sb->strings[0]), &tmp_strings[0], total_length);
    sb->indices = indices;
    return sb;
}

/**
 * Return a copy of string_buffer_t sb in uppercase
 */
string_buffer_ptr dz_upper(string_buffer_t *sb) {
    string_buffer_ptr out_sb;
    int i;
	char c;

    out_sb = dz_string_buffer_create(sb->string_count, sb->char_count);
    for (i = 0; i < sb->char_count; i++) {
		c = sb->strings[i];
        out_sb->strings[i] = ( c >= 97 && c <= 122 ? c - 32 : c );
    }
    memcpy(&(out_sb->indices[0]), &(sb->indices[0]), sb->string_count * sizeof(size_t));
    return out_sb;
}

/**
 * Concatenate two string_buffer_t sb1 and sb2 and return the result
 */
/*
string_buffer_ptr dz_concat(string_buffer_ptr sb1, string_buffer_ptr sb2) {
    int i;
    size_t i1, i2, start1, start2, len1, len2, res_start, res_index;
    string_buffer_ptr res_sb;

    res_sb = dz_string_buffer_create(sb1->string_count, sb1->char_count + sb2->char_count - sb1->string_count);
    res_index = 0;

    for (i = 0; i < sb1->string_count ; i++) {
        i1 = sb1->indices[i];
        start1 = i1 >> (size_t)24;
        len1 = i1 & 0x0000000000FFFFFF;

        i2 = sb2->indices[i];
        start2 = i2 >> (size_t)24;
        len2 = i2 & 0x0000000000FFFFFF;

		memcpy(&(res_sb->strings[res_index]), &(sb1->strings[start1]), len1);
		res_index += len1;

		memcpy(&(res_sb->strings[res_index]), &(sb2->strings[start2]), len2 + 1);
		res_index += len2 + 1;

		res_start = (start1 + start2 - (size_t)i);
		res_sb->indices[i] = (res_start << (size_t)24) + (size_t)(len1 + len2);
	}
	printf("end of dz_concat:");
	dz_string_buffer_print(res_sb);

    return res_sb;
}
*/
expr_node_ptr dz_concat(expr_node_ptr arg1, expr_node_ptr arg2, string_buffer_ptr string_buffers[]) {
    int i;
    size_t i1, i2, start1, start2, len1, len2, res_start, res_index;
    string_buffer_ptr sb1, sb2, res_sb;
	char_ptr s1, s2, res_s;
	expr_node_ptr res_expr_node;

	if (arg1->type == EXPR_NODE_TYPE_STRING_BUFFER_INDEX) {
		sb1 = string_buffers[dz_expr_node_get_string_buffer_index(arg1)];
	}
	else if (arg1->type == EXPR_NODE_TYPE_STRING_CONST) {
		s1 = dz_expr_node_get_string(arg1);
	}
	else {
		printf("Error in dz_concat(): invalid arg1");
		exit(1);
	}

	if (arg2->type == EXPR_NODE_TYPE_STRING_BUFFER_INDEX) {
		sb2 = string_buffers[dz_expr_node_get_string_buffer_index(arg2)];
	}
	else if (arg2->type == EXPR_NODE_TYPE_STRING_CONST) {
		s2 = dz_expr_node_get_string(arg2);
	}
	else {
		printf("Error in dz_concat(): invalid arg2");
		exit(1);
	}

	if (arg1->type == EXPR_NODE_TYPE_STRING_BUFFER_INDEX && arg2->type == EXPR_NODE_TYPE_STRING_BUFFER_INDEX) {
		res_sb = dz_string_buffer_create(sb1->string_count, sb1->char_count + sb2->char_count - sb1->string_count);
		res_index = 0;

		printf("start of dz_concat(sb, ss):");
		dz_string_buffer_print(sb1);
		dz_string_buffer_print(sb2);

		for (i = 0; i < sb1->string_count ; i++) {
			i1 = sb1->indices[i];
			start1 = i1 >> (size_t)24;
			len1 = i1 & 0x0000000000FFFFFF;

			i2 = sb2->indices[i];
			start2 = i2 >> (size_t)24;
			len2 = i2 & 0x0000000000FFFFFF;

			memcpy(&(res_sb->strings[res_index]), &(sb1->strings[start1]), len1);
			res_index += len1;

			memcpy(&(res_sb->strings[res_index]), &(sb2->strings[start2]), len2 + 1);
			res_index += len2 + 1;

			res_start = (start1 + start2 - (size_t)i);
			res_sb->indices[i] = (res_start << (size_t)24) + (size_t)(len1 + len2);
		}
		printf("end of dz_concat(sb, sb):");
		dz_string_buffer_print(res_sb);
		res_expr_node = dz_expr_node_string_buffer_create(res_sb);
	}
	else if (arg1->type == EXPR_NODE_TYPE_STRING_BUFFER_INDEX && arg2->type == EXPR_NODE_TYPE_STRING_CONST) {
		res_sb = dz_string_buffer_create(sb1->string_count, (int)(sb1->char_count + strlen(s2) * sb1->string_count));
		res_index = 0;

		printf("start of dz_concat(sb, s):");
		dz_string_buffer_print(sb1);

		len2 = strlen(s2);
		for (i = 0; i < sb1->string_count ; i++) {
			i1 = sb1->indices[i];
			start1 = i1 >> (size_t)24;
			len1 = i1 & 0x0000000000FFFFFF;

			memcpy(&(res_sb->strings[res_index]), &(sb1->strings[start1]), len1);
			res_index += len1;

			memcpy(&(res_sb->strings[res_index]), s2, len2 + 1);
			res_index += (len2 + 1);

			res_start = (start1 + (size_t)(len2 * i));
			res_sb->indices[i] = (res_start << (size_t)24) + (size_t)(len1 + len2);
		}
		printf("end of dz_concat(sb, s):");
		dz_string_buffer_print(res_sb);
		res_expr_node = dz_expr_node_string_buffer_create(res_sb);
	}
	else if (arg1->type == EXPR_NODE_TYPE_STRING_CONST && arg2->type == EXPR_NODE_TYPE_STRING_BUFFER_INDEX) {
		res_sb = dz_string_buffer_create(sb2->string_count, (int)(sb2->char_count + strlen(s1) * sb2->string_count));
		res_index = 0;

		printf("start of dz_concat(s, sb):");
		dz_string_buffer_print(sb2);

		len1 = strlen(s1);
		for (i = 0; i < sb2->string_count ; i++) {
			i2 = sb2->indices[i];
			start2 = i2 >> (size_t)24;
			len2 = i2 & 0x0000000000FFFFFF;

			memcpy(&(res_sb->strings[res_index]), s1, len1);
			res_index += len1;

			memcpy(&(res_sb->strings[res_index]), &(sb2->strings[start2]), len2 + 1);
			res_index += (len2 + 1);

			res_start = (size_t)(len1 * i + start2);
			res_sb->indices[i] = (res_start << (size_t)24) + (size_t)(len1 + len2);
		}
		printf("end of dz_concat(s, sb):");
		dz_string_buffer_print(res_sb);
		res_expr_node = dz_expr_node_string_buffer_create(res_sb);
	}
	else {
		res_s = (char*) malloc(strlen(s1) + strlen(s2) + 1);
		res_expr_node = dz_expr_node_string_create(res_s);
	}

	return res_expr_node;
}
/**
 * Evaluate an expr_node_t n and return the result
 */
expr_node_ptr dz_eval(expr_node_ptr n, string_buffer_ptr string_buffers[]) {
	expr_node_ptr arg1, arg2;

    if (n->type == EXPR_NODE_TYPE_CONCAT) {
		arg1 = dz_expr_node_get_arg(n, 0);
		//sb1 = string_buffers[dz_expr_node_get_string_buffer_index(n1)];
		arg2 = dz_expr_node_get_arg(n, 1);
        //sb2 = string_buffers[dz_expr_node_get_string_buffer_index(n2)];
        return dz_concat(arg1, arg2, string_buffers);
    }
/*
    else if (n->type == EXPR_NODE_TYPE_UPPER) {
		arg1 = dz_expr_node_get_arg(n, 0);
		//sb1 = string_buffers[dz_expr_node_get_string_buffer_index(n1)];
        return dz_upper(arg1);
    }
*/
	else {
		printf("Error in dz_eval()");
		exit(1);
	}

    return NULL;
}

/**
 * Test for concat(string_buffer, string_buffer) 
 */
expr_node_ptr test_concat1() {
	expr_node_ptr node1 = dz_expr_node_string_buffer_index_create(0);
    expr_node_ptr node2 = dz_expr_node_string_buffer_index_create(1);
    expr_node_ptr node3 = dz_expr_node_create(EXPR_NODE_TYPE_CONCAT, node1, node2, 0);

    return node3;
}

/**
 * Test for concat(string_buffer, string) 
 */
expr_node_ptr test_concat2() {
	expr_node_ptr node1 = dz_expr_node_string_buffer_index_create(0);
    expr_node_ptr node2 = dz_expr_node_string_create("OK");
    expr_node_ptr node3 = dz_expr_node_create(EXPR_NODE_TYPE_CONCAT, node1, node2, 0);

    return node3;
}

/**
 * Test for concat(string, string_buffer) 
 */
expr_node_ptr test_concat3() {
	expr_node_ptr node1 = dz_expr_node_string_buffer_index_create(0);
    expr_node_ptr node2 = dz_expr_node_string_create("OK");
    expr_node_ptr node3 = dz_expr_node_create(EXPR_NODE_TYPE_CONCAT, node2, node1, 0);

    return node3;
}

/**
 * Test for upper() 
 */
expr_node_ptr test_upper() {
	expr_node_ptr node1 = dz_expr_node_string_buffer_index_create(0);
	expr_node_ptr node2 = dz_expr_node_create(EXPR_NODE_TYPE_UPPER, node1, 0, 0);

    return node2;
}

int main(int argc, char **argv ) {
    expr_node_ptr t, res;
	string_buffer_ptr res_sb, sb_list[2];

    sb_list[0] = dz_string_buffer_make_random(30);
    sb_list[1] = dz_string_buffer_make_random(30);

    t = test_concat3();
    res = dz_eval(t, sb_list);
	res_sb = dz_expr_node_get_string_buffer(res);
    dz_string_buffer_print(res_sb);

    return 0;
}

