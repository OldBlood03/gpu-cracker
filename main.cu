#include <cstdio>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <stddef.h>
#include <stdbool.h>
#include <assert.h>
#include <stdint.h>
#include <malloc.h>
#include <string.h>
#include <stddef.h>
#include <stdio.h>
#include <unistd.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define SHA256_DIGEST_LENGTH 32
#define HASH_BYTE_LENGTH 16
#define HASH_STRING_LENGTH 16
#define SALT_LENGTH 17
#define PRESALT_LENGTH 12
#define CHARACTER_LIMIT 6
#define TOTAL_ATTEMPT_STRING_LENGTH (SALT_LENGTH + PRESALT_LENGTH + CHARACTER_LIMIT + 1)

#define MAX_GPU_THREADS_IN_BLOCK 1024
#define MAX_GPU_BLOCKS 65535

#define CHUNK_BYTE_LENGTH 64 


#define ROTR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
#define SHR(x, n)  ((x) >> (n))

#define CH(x, y, z)  (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x, y, z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define SIGMA0(x)    (ROTR(x, 2) ^ ROTR(x, 13) ^ ROTR(x, 22))
#define SIGMA1(x)    (ROTR(x, 6) ^ ROTR(x, 11) ^ ROTR(x, 25))
#define sigma0(x)    (ROTR(x, 7) ^ ROTR(x, 18) ^ SHR(x, 3))
#define sigma1(x)    (ROTR(x, 17) ^ ROTR(x, 19) ^ SHR(x, 10))

__device__ static const char basis[] = "qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM!\"#%&/()=?1234567890";
__device__ static const int radix = 73;
__device__ static const char presalt[] = "potPlantSalt";

void bitify(char hex[HASH_STRING_LENGTH], unsigned char out[HASH_BYTE_LENGTH]){
    for (int i = 0, hi = 0; i < 16 && hi < 32; i++, hi +=2) {
        out[i] = 0;
        switch (hex[hi]) {

            case '0' ... '9':
                out[i] |= (hex[hi]-'0') << 4;
                break;
            case 'a' ... 'f':
                out[i] |= (hex[hi]-'a' + 10) << 4;
                break;
            default:
                printf("Error");
        }

        switch (hex[hi + 1]) {
            case '0' ... '9':
                out[i] |= (hex[hi + 1]-'0');
                break;
            case 'a' ... 'f':
                out[i] |= (hex[hi + 1]-'a' + 10);
                break;
            default:
                printf("Error");
        }
    }
}


__device__ unsigned char *gpu_sha_256 (const unsigned char *message, unsigned int message_length, unsigned char store[HASH_BYTE_LENGTH]){
    static_assert(TOTAL_ATTEMPT_STRING_LENGTH*8< (512 - 64 - 8), 
            "The possible length of the string is more than what the sha algorithm is meant to handle");

    const uint32_t K[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
    };
    uint32_t H[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };

    unsigned char chunk[CHUNK_BYTE_LENGTH];
    memset(chunk, 0, CHUNK_BYTE_LENGTH);

    // Copy the original message into the chunk.
    memcpy(chunk, message, message_length);

    // Append the '1' bit (0x80) after the message.
    chunk[message_length] = 0x80;

    uint64_t bit_length = (uint64_t)message_length * 8;
    chunk[63] = bit_length & 0xFF;
    chunk[62] = (bit_length >> 8) & 0xFF;
    chunk[61] = (bit_length >> 16) & 0xFF;
    chunk[60] = (bit_length >> 24) & 0xFF;

    uint32_t W[64];
    uint32_t a, b, c, d, e, f, g, h;
    int t;

    for (t = 0; t < 16; ++t) {
        W[t] = ((uint32_t)chunk[t * 4]     << 24) |
               ((uint32_t)chunk[t * 4 + 1] << 16) |
               ((uint32_t)chunk[t * 4 + 2] << 8)  |
               ((uint32_t)chunk[t * 4 + 3]);
    }

    // Extend the first 16 words into the remaining 48 words of the schedule.
    for (t = 16; t < 64; ++t) {
        W[t] = sigma1(W[t - 2]) + W[t - 7] + sigma0(W[t - 15]) + W[t - 16];
    }

    // b) Initialize working variables with the initial hash values.
    a = H[0]; b = H[1]; c = H[2]; d = H[3];
    e = H[4]; f = H[5]; g = H[6]; h = H[7];

    // c) Compression loop (64 rounds).
    for (t = 0; t < 64; ++t) {
        uint32_t T1 = h + SIGMA1(e) + CH(e, f, g) + K[t] + W[t];
        uint32_t T2 = SIGMA0(a) + MAJ(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + T1;
        d = c;
        c = b;
        b = a;
        a = T1 + T2;
    }

    H[0] += a; H[1] += b; H[2] += c; H[3] += d;
    H[4] += e; H[5] += f; H[6] += g; H[7] += h;
    
    for (int i = 0; i < 8; ++i) {
        store[i * 4]     = (H[i] >> 24) & 0xFF;
        store[i * 4 + 1] = (H[i] >> 16) & 0xFF;
        store[i * 4 + 2] = (H[i] >> 8) & 0xFF;
        store[i * 4 + 3] = H[i] & 0xFF;
    }

    return store;
}

__global__ void gpu_compute_password (const char* salt, const unsigned char *compare_hash, size_t offset, char *found){

    const size_t thread = threadIdx.x;
    const size_t block_stride = blockDim.x;
    const size_t block = blockIdx.x;
    const size_t id = thread + block*block_stride + offset;

    char string[SALT_LENGTH + PRESALT_LENGTH + CHARACTER_LIMIT] = {0};

    for (int i = 0; i < PRESALT_LENGTH; i++){
        string[i] = presalt[i];
    }
    int attempt_length = 0;
    for (size_t i = id; i != 0; i /= radix){
        string[PRESALT_LENGTH + attempt_length++] = basis[i%radix];
    }
    for (int i = 0; i < SALT_LENGTH; i++) {
        string[PRESALT_LENGTH + attempt_length + i] = salt[i];
    }
    const char paswd[] = "zf17kFAa";
    bool the_attempt = false;
    if (attempt_length == 7){
        printf("hello\n");
        for (int i = 0; i < attempt_length; i++){
            if (paswd[i] != string[PRESALT_LENGTH + i]){
                goto FAILED;
            }
        }
        the_attempt = true;
    }
FAILED:

    const int total_length = PRESALT_LENGTH + SALT_LENGTH + attempt_length;
    unsigned char hash[SHA256_DIGEST_LENGTH];

    gpu_sha_256((const unsigned char*) string, total_length, hash);
    if (the_attempt){
        printf("\033[2Khash computed: ");
        for (int i = 0; i < SHA256_DIGEST_LENGTH; i++) {
            printf("%x", hash[i]);
        }
        printf("\n");
        printf("\033[2Khash reference: ");
        for (int i = 0; i < SHA256_DIGEST_LENGTH; i++) {
            printf("%x", compare_hash[i]);
        }
        printf("\n");
    }

    for (int i = 0; i < HASH_BYTE_LENGTH; i++) {
        if (hash[i] != compare_hash[i]){
            return;
        }
    }
    for (int i = 0; i < attempt_length; i++) {
        found[i] = string[PRESALT_LENGTH + i];
    }
    found[attempt_length] = '\0';
}

#define BAR_WIDTH 40
void print_progress (size_t current, size_t out_of){
    const char bar[BAR_WIDTH] = "=======================================";
    const char empty[BAR_WIDTH] = "                                       ";
    const float percentage = (float)current/out_of;
    const int current_width = percentage*BAR_WIDTH;
    const int remaining_width = BAR_WIDTH - current_width;
    printf("\033[2K\r[%.*s%.*s]%.3f", current_width, bar,remaining_width, empty, percentage);
    fflush(stdout);
}

int main(int argc, char *argv[])
{
    size_t max_iterations = radix;
    for (int i = 0; i < CHARACTER_LIMIT-1; i++) {
        max_iterations *= radix;
    }

    const char feedfile[] = "feed.txt";
    FILE *feedptr = fopen(feedfile, "r");
    assert(feedptr);

    size_t feed_size = 0;
    char *feed = NULL;

    char *found;
    gpuErrchk(cudaMalloc(&found, sizeof(char[CHARACTER_LIMIT+1])));

    while (getline(&feed, &feed_size, feedptr) != -1) {
        char *tokker = feed;
        char *name = strtok(tokker, ",");

        char *salt = strtok(NULL, ",");
        char *feed_hash = strtok(NULL, ",");
        unsigned char feed_bits[HASH_BYTE_LENGTH];
        assert(name && salt && feed_hash);
        bitify(feed_hash, feed_bits);
        assert(feed_bits);
        printf("\n\033[2KWorking on %s\n", name);
        
        char *cuda_salt, *cuda_feed_bits;
        gpuErrchk(cudaMalloc(&cuda_salt, sizeof(char[SALT_LENGTH+1])));
        gpuErrchk(cudaMalloc(&cuda_feed_bits, sizeof(unsigned char[HASH_BYTE_LENGTH])));

        char value[CHARACTER_LIMIT + 1] = {0};
        gpuErrchk(cudaMemset(found, 0, sizeof(char[CHARACTER_LIMIT+1])));
        bool stop = false;
        for (size_t i = 0; i < max_iterations && !stop; i += MAX_GPU_BLOCKS*MAX_GPU_THREADS_IN_BLOCK) {
            gpu_compute_password<<<MAX_GPU_BLOCKS,MAX_GPU_THREADS_IN_BLOCK>>>(cuda_salt, (const unsigned char *)cuda_feed_bits, i, found);
            gpuErrchk(cudaGetLastError());
            gpuErrchk(cudaMemcpy(value, found, sizeof(char[CHARACTER_LIMIT + 1]), cudaMemcpyDeviceToHost));

            print_progress(i, max_iterations);
            if(*value){
                printf("\033[2K\rFOUND PASSWORD: %s FOR PERSON %s", value, name);
                fflush(stdout);
                stop = true;
            }
        }

    }

    return 0;
}

