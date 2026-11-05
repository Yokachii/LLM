#ifndef DATASET_H
#define DATASET_H

typedef struct {
    int V;                      // Nombre de caractère
    int *dic;                  // dict[char] -> indice dans map
    char *dic_inv;             // dic_inv[indice] -> char
    float **map;               // caractère -> indice, -1 si absent
} Vocab;

typedef struct {
    int n;                      // nombre de tokens
    unsigned char *tok;         // n indices
} Texte;

typedef struct
{
    int TP;
    int TN;
    int FN;
    int FP;
} MatrixConfusion;

Texte *new_text(int n, Vocab *voc);

void free_text(Texte *t);
void free_vocab(Vocab *v);

Texte *load_texte(char *path, Vocab **v, int d);


void display_Text(Texte t);
void display_Vocab(Vocab v);


float sigmoide(float z);

// float tanh(float z);

void split_texte(Texte *t, Texte **train, Texte **test, float ratio);

#endif