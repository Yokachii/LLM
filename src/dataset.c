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

Texte *load_texte(char *path, Vocab **v, int d){

    FILE *fichier = fopen(path, "r");
    if (fichier == NULL) {
        printf("Erreur lors du chargement\n");
        return NULL;
    }
    
    long n = getFileSize(fichier);

    unsigned char *fileContents = malloc(n+1);

    fread(fileContents, 1, n, fichier);

    fileContents[n] = '\0'; // Ajout du caractère de fin

    fclose(fichier);


    int V = 0;
    *v = malloc(sizeof(Vocab));
    int *dic = malloc(sizeof(int)*512);
    unsigned char *dic_inv = malloc(sizeof(char)*512);

    for(int i = 0; i<512; i++) dic[i]=-1;

    for(long i = 0; i<n;i++){
        unsigned char cc = fileContents[i];
        if(dic[cc]==-1){
            dic[cc] = V;
            dic_inv[V] = cc;
            V++;
        }
    }

    Texte *new_t = malloc(sizeof(Texte));
    new_t->n=n;
    new_t->tok=(int *)malloc(n * sizeof(int));
    (*v)->V=V;
    float **vocabulaire = malloc(sizeof(*vocabulaire)*V);
    for(int i = 0; i<V; i++){
        float *ligne = malloc(d*sizeof(float));
        for(int j = 0; j<d; j++){
            ligne[j] = (float)rand() / RAND_MAX - 0.5f;
        }
        vocabulaire[i]=ligne;
    }
    new_t->voc=*v;

    for(long i = 0; i<n; i++){
        new_t->tok[i]=dic[fileContents[i]];
        // printf("%d ",new_t->tok[i]);
    }

    (*v)->emb=vocabulaire;
    (*v)->dic=dic;
    (*v)->d=d;
    (*v)->dic_inv=dic_inv;

    free(fileContents);

    return new_t;
};

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