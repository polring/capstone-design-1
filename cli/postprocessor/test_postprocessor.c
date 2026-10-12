/**
 * @file test_postprocessor.c
 * @brief Unit tests in C matching Python tests/test_postprocessor_python.py.
 */

#include "postprocessor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_tests_passed = 0;
static int g_tests_total = 0;

#define ASSERT_TRUE(expr, msg) do { \
    g_tests_total++; \
    if (expr) { \
        g_tests_passed++; \
        printf("  [PASS] %s\n", msg); \
    } else { \
        printf("  [FAIL] %s (Line %d: %s)\n", msg, __LINE__, #expr); \
    } \
} while(0)

static const char *TEST_DICTIONARY[] = {
    "users",
    "user_name",
    "email",
    "id",
    "name",
    "department",
    "manager_id"
};
static const size_t TEST_DICT_SIZE =
    sizeof(TEST_DICTIONARY) / sizeof(TEST_DICTIONARY[0]);

void test_pre_correct_keywords(void) {
    printf("Running test_pre_correct_keywords...\n");
    const char *sql = "SELETC id, name FRMO users";
    char *fixed = pre_correct_keywords(sql);
    ASSERT_TRUE(fixed != NULL, "pre_correct_keywords returns non-null");
    if (fixed) {
        ASSERT_TRUE(strcmp(fixed, "SELECT id, name FROM users") == 0,
                    "SELETC -> SELECT and FRMO -> FROM corrected");
        free(fixed);
    }
}

void test_typo_correction_keywords(void) {
    printf("Running test_typo_correction_keywords...\n");
    const char *sql = "SELETC id, name FRMO users";
    char *result = process_sql(sql, "typo", TEST_DICTIONARY, TEST_DICT_SIZE, 0, 0.0);
    ASSERT_TRUE(result != NULL, "process_sql returns non-null");
    if (result) {
        ASSERT_TRUE(strcmp(result, "SELECT id, name FROM users") == 0,
                    "Full query with keyword typos resolved accurately");
        free(result);
    }
}

void test_typo_correction_identifiers(void) {
    printf("Running test_typo_correction_identifiers...\n");
    const char *sql = "SELECT usr_name, emil FROM users";
    char *result = process_sql(sql, "typo", TEST_DICTIONARY, TEST_DICT_SIZE, 0, 0.0);
    ASSERT_TRUE(result != NULL, "process_sql returns non-null");
    if (result) {
        ASSERT_TRUE(strstr(result, "user_name") != NULL, "usr_name corrected to user_name");
        ASSERT_TRUE(strstr(result, "email") != NULL, "emil corrected to email");
        free(result);
    }
}

void test_typo_correction_complex(void) {
    printf("Running test_typo_correction_complex...\n");
    const char *sql = "SELECT manger_id FROM departmnt WHERE id = 1";
    char *result = process_sql(sql, "typo", TEST_DICTIONARY, TEST_DICT_SIZE, 0, 0.0);
    ASSERT_TRUE(result != NULL, "process_sql returns non-null");
    if (result) {
        ASSERT_TRUE(strstr(result, "manager_id") != NULL, "manger_id corrected to manager_id");
        ASSERT_TRUE(strstr(result, "department") != NULL, "departmnt corrected to department");
        free(result);
    }
}

void test_ngram_correction(void) {
    printf("Running test_ngram_correction...\n");
    const char *sql = "SELECT id FROM departmnt";
    char *result = process_sql(sql, "ngram", TEST_DICTIONARY, TEST_DICT_SIZE, 0, 0.0);
    ASSERT_TRUE(result != NULL, "process_sql returns non-null");
    if (result) {
        ASSERT_TRUE(strstr(result, "department") != NULL, "departmnt corrected to department via ngram");
        free(result);
    }
}

void test_strategy_chaining(void) {
    printf("Running test_strategy_chaining...\n");
    const char *sql = "SELECT usr_name FROM departmnt";
    char *result = process_sql(sql, "typo,ngram", TEST_DICTIONARY, TEST_DICT_SIZE, 0, 0.0);
    ASSERT_TRUE(result != NULL, "process_sql returns non-null");
    if (result) {
        ASSERT_TRUE(strstr(result, "user_name") != NULL, "usr_name corrected in chained strategy");
        ASSERT_TRUE(strstr(result, "department") != NULL, "departmnt corrected in chained strategy");
        free(result);
    }
}

void test_semantic_correction_pipeline(void) {
    printf("Running test_semantic_correction_pipeline...\n");
    const char *sql = "SELECT usr FROM users";
    char *result = process_sql(sql, "semantic", TEST_DICTIONARY, TEST_DICT_SIZE, 300, 0.0);
    ASSERT_TRUE(result != NULL, "process_sql returns non-null");
    if (result) {
        ASSERT_TRUE(strstr(result, "usr") != NULL || strstr(result, "user_name") != NULL,
                    "semantic fallback or match safely handles query");
        free(result);
    }
}

int main(void) {
    printf("=========================================\n");
    printf("Running C SQL Post-processor Unit Tests\n");
    printf("=========================================\n\n");

    test_pre_correct_keywords();
    test_typo_correction_keywords();
    test_typo_correction_identifiers();
    test_typo_correction_complex();
    test_ngram_correction();
    test_strategy_chaining();
    test_semantic_correction_pipeline();

    printf("\n=========================================\n");
    printf("Summary: %d / %d assertions passed.\n", g_tests_passed, g_tests_total);
    printf("=========================================\n");

    return (g_tests_passed == g_tests_total) ? 0 : 1;
}
