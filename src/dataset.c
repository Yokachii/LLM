#include "../include/dataset.h"
#include "../include/perceptron.h"
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define Max_Char 1024

long getFileSize(FILE* fp)
{
   long size = 0;
   fpos_t pos;
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

    if (v->map != NULL) {
        for (int i = 0; i < v->V; i++) {
            free(v->map[i]);
        }
    }
    free(v->map);
    free(v->dic);
    free(v->dic_inv);
    free(v);
}

void display_Text(Texte t){
    if (t.n < 0 || (t.n > 0 && t.tok == NULL)) {
        fprintf(stderr, "Erreur : texte invalide\n");
        return;
    }

    printf("]\n");
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
    
    /* Determine the size of the file */
    long n = getFileSize(fichier);

    /* Allocate memory for the contents */
    char *fileContents = malloc(n+1);

    /* Read the contents */
    fread(fileContents, 1, n, fichier);

    /* fread does not automatically add a terminating NULL character.
       You must add it yourself. */
    fileContents[n] = '\0';

    fclose(fichier);


    int V = 0;
    *v = malloc(sizeof(Vocab));
    int *dic = malloc(sizeof(int)*512);
    char *dic_inv = malloc(sizeof(char)*512);

    for(int i = 0; i<512; i++) dic[i]=-1;

    for(int i = 0; i<n/sizeof(char);i++){
        char cc = fileContents[i];
        if(dic[cc]==-1){
            dic[cc] = i;
            dic_inv[i] = cc;
            V++;
        }
    }


    Texte *new_text = malloc(sizeof(Texte));
    new_text->n=n;
    new_text->tok=(unsigned char *)fileContents;
    (*v)->V=V;
    float **vocabulaire = malloc(sizeof(*vocabulaire)*V);
    for(int i = 0; i<V; i++){
        float *ligne = malloc(d*sizeof(float));
        for(int j = 0; j<d; j++){
            ligne[j]=0;
        }
        vocabulaire[i]=ligne;
    }

    (*v)->map=vocabulaire;
    (*v)->dic=dic;
    (*v)->dic_inv=dic_inv;

    return new_text;
};

float sigmoide(float z){
    return (1 / (1 + exp(-z)));
}

// float tanh(float z){
//     return z;
// }


void split_texte(Texte *t, Texte **train, Texte **test, float ratio){
//     int dim = d->dim;
//     int n = d->n;
//     int nTrain = n*ratio;
//     int nTest = n - nTrain;
    
//     Dataset *resTrain = new_dataset(dim,nTrain);
//     Dataset *resTest = new_dataset(dim,nTest);
//     for(int i = 0; i<n;i++){
//         if(i<nTrain){
//             for(int j= 0; j<dim; j++){
//                 resTrain->data[i][j]=d->data[i][j];
//             }
//         }else{
//             for(int j= 0; j<dim; j++){
//                 resTest->data[i-nTrain][j]=d->data[i][j];
//             }
//         }
//     }
//     *train=resTrain;
//     *test=resTest;
//     free_dataset(d);
}