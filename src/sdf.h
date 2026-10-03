#pragma once

#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/euler_angles.hpp>


// OPERATIONS
__host__ __device__ inline float smoothMin(float a, float b, float k) {
    float h = glm::max(k - glm::abs(a - b), 0.f) / k;
    return glm::min(a, b) - h * h * k * 0.25f;
}

// PRIMITIVES
__host__ __device__ inline float sdfSphere(glm::vec3 query, float radius) {
    return glm::length(query) - radius;
}

__host__ __device__ inline float sdfBox(glm::vec3 query, glm::vec3 halfExtents) {
    glm::vec3 q = glm::abs(query) - halfExtents;
    return glm::length(glm::max(q, 0.f)) + glm::min(glm::max(q.x, glm::max(q.y,q.z)), 0.f);
}

__host__ __device__ inline float sdfMetaballs(glm::vec3 query, float k) {
    float s1 = sdfSphere(query - glm::vec3(0.f), 2.f);
    float s2 = sdfSphere(query - glm::vec3(2.f, 0.f, 1.f), 1.f);
    float s3 = sdfSphere(query - glm::vec3(-2.f, 2.f, 0.f), 1.5f);

    return smoothMin(s1, smoothMin(s2, s3, k), k);
}

__host__ __device__ inline float sdfCross(glm::vec3 query) {
    float da = sdfBox(glm::vec3(query.x, query.y,     0.f), glm::vec3(1.f));
    float db = sdfBox(glm::vec3(    0.f, query.y, query.z), glm::vec3(1.f));
    float dc = sdfBox(glm::vec3(query.z,     0.f, query.x), glm::vec3(1.f));
    return glm::min(da, glm::min(db,dc));
}

//https://iquilezles.org/articles/menger/
#define MENGER_ITERATIONS 5
__host__ __device__ inline float sdfMenger(glm::vec3 query) {
    float d = sdfBox(query, glm::vec3(1.f));
    glm::mat3 rotate = glm::orientate3(glm::radians(glm::vec3(15.f, 35.f, 50.f)));


    float s = 1.f;
    for (int m=0; m<MENGER_ITERATIONS; m++) {
        // Mess w/ iteration
        query += glm::vec3(0.1f);
        query = query * rotate;
        
        
        glm::vec3 a = glm::mod(query * s, 2.f) - 1.f;
        s *= 3.f;
        glm::vec3 r = 1.f - 3.f * glm::abs(a);

        float c = sdfCross(r) / s;
        d = glm::max(d, c);
    }
    return d;
}

__host__ __device__ inline float sceneSdf(glm::vec3 query, GeomType type) {
    if (type == SDF_SPHERE)
        return sdfSphere(query, 0.5f);
    else if (type == SDF_CUBE)
        return sdfBox(query, glm::vec3(0.5f));
    else if (type == SDF_METABALLS)
        return sdfMetaballs(query, 0.5f);
    else if (type == SDF_MENGER)
        return sdfMenger(query);
    else
        return INFINITY;
}

__host__ __device__ inline glm::vec3 sdfNormal(glm::vec3 query, GeomType type) {
    float e = 0.001f;
    // Avg tetrahedron (from IQ, faster than 6 dir avg)
    const glm::vec3 k0 = glm::vec3( 1, -1, -1);
    const glm::vec3 k1 = glm::vec3(-1, -1,  1);
    const glm::vec3 k2 = glm::vec3(-1,  1, -1);
    const glm::vec3 k3 = glm::vec3( 1,  1,  1);
    return glm::normalize(
        k0 * sceneSdf(query + e * k0, type) +
        k1 * sceneSdf(query + e * k1, type) +
        k2 * sceneSdf(query + e * k2, type) +
        k3 * sceneSdf(query + e * k3, type)
    );
}
