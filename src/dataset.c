#include "../include/dataset.h"
#include "../include/perceptron.h"
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

long getFileSize(FILE* fp)
{
   long size = 0;
   fseek(fp, 0, SEEK_END);
   size = ftell(fp);
   fseek(fp, 0, SEEK_SET);
   return size;
}

void free_text(Texte *t){
    if (t == NULL) return;

    free(t->tok);
    free(t);
}

void free_vocab(Vocab *v){
    if (v == NULL) return;

    if (v->emb != NULL) {
        for (int i = 0; i < v->V; i++) {
            free(v->emb[i]);
        }
    }
    free(v->emb);
    free(v->dic);
    free(v->dic_inv);
    free(v);
}

void display_Text(Texte t){
    if (t.n < 0 || (t.n > 0 && t.tok == NULL)) {
        fprintf(stderr, "Erreur : texte invalide\n");
        return;
    }

    for(int i = 0; i < t.n; i++) {
        unsigned char c = (unsigned char)(t.voc->dic_inv[t.tok[i]]);
        if (isprint(c)) {
            putchar(c);
        } else {
            switch (c) {
                case '\n':
                    printf("\\n");
                    break;
                case '\r':
                    printf("\\r");
                    break;
                case '\t':
                    printf("\\t");
                    break;
                case '\0':
                    printf("\\0");
                    break;
                default:
                    printf("\\x%02X", (unsigned int)c);
                    break;
            }
        }
    }
}

void display_Vocab(Vocab v){
    if (v.V < 0 || (v.V > 0 && (v.dic == NULL || v.dic_inv == NULL))) {
        fprintf(stderr, "Erreur : vocabulaire invalide\n");
        return;
    }

    printf("Vocabulaire (%d caractères) :\n", v.V);
    for (int i = 0; i < v.V; i++) {
        unsigned char c = (unsigned char)v.dic_inv[i];

        printf("  %d : ", i);
        if (isprint(c)) {
            printf("'%c'", c);
        } else {
            switch (c) {
                case '\n':
                    printf("'\\n'");
                    break;
                case '\r':
                    printf("'\\r'");
                    break;
                case '\t':
                    printf("'\\t'");
                    break;
                case '\0':
                    printf("'\\0'");
                    break;
                default:
                    printf("'\\x%02X'", (unsigned int)c);
                    break;
            }
        }
        printf(" (dic[%u] = %d)\n", (unsigned int)c, v.dic[c]);
    }
}

static unsigned char *lire_fichier(const char *path, long *n){
    FILE *f = fopen(path, "rb");
    if (f == NULL) {
        fprintf(stderr, "Erreur : impossible d'ouvrir '%s'\n", path);
        return NULL;
    }
    long size = getFileSize(f);
    unsigned char *buf = malloc((size_t)size + 1);
    if (buf == NULL || fread(buf, 1, (size_t)size, f) != (size_t)size) {
        fprintf(stderr, "Erreur : lecture de '%s'\n", path);
        free(buf);
        fclose(f);
        return NULL;
    }
    fclose(f);
    buf[size] = '\0';
    *n = size;
    return buf;
}

Vocab *construire_vocab(const char *path, int d){
    long n = 0;
    unsigned char *buf = lire_fichier(path, &n);
    if (buf == NULL) return NULL;

    int *dic = malloc(sizeof(int) * 256);
    unsigned char *dic_inv = malloc(256);
    for (int i = 0; i < 256; i++) dic[i] = -1;

    int V = 0;
    for (long i = 0; i < n; i++) {
        unsigned char c = buf[i];
        if (dic[c] == -1) {
            dic[c] = V;
            dic_inv[V] = c;
            V++;
        }
    }
    free(buf);

    Vocab *voc = malloc(sizeof(Vocab));
    voc->V = V;
    voc->d = d;
    voc->dic = dic;
    voc->dic_inv = realloc(dic_inv, (size_t)V);
    voc->emb = malloc(sizeof(float *) * (size_t)V);
    for (int i = 0; i < V; i++) {
        voc->emb[i] = malloc(sizeof(float) * (size_t)d);
        for (int j = 0; j < d; j++)
            voc->emb[i][j] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    }
    return voc;
}

Texte *encoder_texte(const char *path, Vocab *voc){
    long n = 0;
    unsigned char *buf = lire_fichier(path, &n);
    if (buf == NULL) return NULL;

    Texte *t = malloc(sizeof(Texte));
    t->tok = malloc((size_t)n * sizeof(int));
    t->voc = voc;

    long k = 0, inconnus = 0;
    for (long i = 0; i < n; i++) {
        int id = voc->dic[buf[i]];
        if (id >= 0) t->tok[k++] = id;
        else inconnus++;
    }
    t->n = (int)k;
    free(buf);

    if (inconnus > 0)
        fprintf(stderr, "Attention : %ld caractères hors vocabulaire ignorés\n", inconnus);
    return t;
}

Texte *load_texte(char *path, Vocab **v, int d){
    *v = construire_vocab(path, d);
    if (*v == NULL) return NULL;
    return encoder_texte(path, *v);
}

float sigmoide(float z){
    return (1 / (1 + exp(-z)));
}


void split_texte(Texte *t, Texte **train, Texte **test, float ratio){
    if (train == NULL || test == NULL) {
        fprintf(stderr, "Erreur : sorties invalides pour split_texte\n");
        return;
    }

    *train = NULL;
    *test = NULL;

    // if (t == NULL || t->n < 0 || (t->n > 0 && t->tok == NULL) ||
    //     !isfinite(ratio) || ratio < 0.0f || ratio > 1.0f) {
    //     fprintf(stderr, "Erreur : texte ou ratio invalide pour split_texte\n");
    //     return;
    // }

    int nTrain = (int)(t->n * ratio);
    int nTest = t->n - nTrain;
    Texte *resTrain = malloc(sizeof(*resTrain));
    Texte *resTest = malloc(sizeof(*resTest));
    if (resTrain == NULL || resTest == NULL) {
        fprintf(stderr, "Erreur : allocation impossible dans split_texte\n");
        free(resTrain);
        free(resTest);
        return;
    }

    resTrain->n = nTrain;
    resTrain->voc = t->voc;
    resTrain->tok = nTrain > 0 ? malloc((size_t)nTrain * sizeof(*resTrain->tok)) : NULL;
    resTest->n = nTest;
    resTest->voc = t->voc;
    resTest->tok = nTest > 0 ? malloc((size_t)nTest * sizeof(*resTest->tok)) : NULL;

    if ((nTrain > 0 && resTrain->tok == NULL) ||
        (nTest > 0 && resTest->tok == NULL)) {
        fprintf(stderr, "Erreur : allocation impossible dans split_texte\n");
        free_text(resTrain);
        free_text(resTest);
        return;
    }

    if (nTrain > 0) {
        memcpy(resTrain->tok, t->tok, (size_t)nTrain * sizeof(*t->tok));
    }
    if (nTest > 0) {
        memcpy(resTest->tok, t->tok + nTrain, (size_t)nTest * sizeof(*t->tok));
    }

    *train = resTrain;
    *test = resTest;
    free_text(t);
}