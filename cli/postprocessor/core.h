/**
 * @file core.h
 * @brief Core postprocessor coordination.
 */

#ifndef POSTPROCESSOR_CORE_H
#define POSTPROCESSOR_CORE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

char *process_sql(
    const char *sql_query,
    const char *strategy,
    const char **dictionary,
    size_t dict_size,
    int dim,
    double threshold
);

#ifdef __cplusplus
}
#endif

#endif /* POSTPROCESSOR_CORE_H */
