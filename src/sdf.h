#pragma once

#include <cuda_runtime.h>
#include <glm/glm.hpp>

__host__ __device__ inline float sdfSphere(glm::vec3 query, float radius) {
    return glm::length(query) - radius;
}

__host__ __device__ inline float sdfBox(glm::vec3 query, glm::vec3 halfExtents) {
    glm::vec3 q = glm::abs(query) - halfExtents;
    return glm::length(glm::max(q, 0.f)) + glm::min(glm::max(q.x, glm::max(q.y,q.z)), 0.f);
}

__host__ __device__ inline float sceneSdf(glm::vec3 query, GeomType type) {
    if (type == SDF_SPHERE) {
        return sdfSphere(query, 0.5f);
    }
    else if (type == SDF_CUBE) {
        return sdfBox(query, glm::vec3(0.5f));
    } else {
        return INFINITY;
    }
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