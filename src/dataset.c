#include "../include/dataset.h"
#include "../include/perceptron.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>


/**
 * \fn int new_dataset(int dim, int taux) 
 * \brief Initialise un dataset
 *
 * \param dim nombre de wi xi 
 * \param n nombre de données 
 * \return pointeur vers le nouveau dataset
 */ 
Dataset *new_dataset(int dim, int n){
    Dataset *n_dataset = malloc(sizeof(Dataset));
    n_dataset->n=n;
    n_dataset->dim=dim;
    float **data = malloc(n*sizeof(float*));
    for(int i = 0; i<n; i++){
        float *ligne = malloc(dim*sizeof(float));
        for(int j = 0; j<dim; j++){
            ligne[j]=0;
        }
        data[i]=ligne;
    }
    n_dataset->data=data;
    return n_dataset;
};

/**
 * \fn void *free_dataset(Dataset *p)
 * \brief Libère un dataset
 *
 * \param p Pointeur vers le dataset à free 
 */ 
void *free_dataset(Dataset *d){
    for(int i = 0; i<d->n; i++){
        free(d->data[i]);
    }
    free(d->data);
    free(d);
};

/**
 * \fn void *display_dataset(dataset p)
 * \brief Affiche un dataset
 *
 * \param p dataset à afficher
 */ 
void display_dataset(Dataset d){
    int n = d.n;
    int dim = d.dim;
    float **data = d.data;
    printf("N : %d\nDim : %d\n[\n",n,dim);
    for(int i = 0; i<n; i++){
        printf("    [");
        for(int j = 0; j<dim; j++){
            printf("%.2f, ",data[i][j]);
        }
        printf("]\n");
    }
    printf("]\n");
};

/**
 * \fn Dataset *load_dataset(char *path)
 * \brief Charge un dataset
 *
 * \param path local path
 */ 
Dataset *load_dataset(char *path, int d, int *V, float ***dict){

    // Lecture de n & dim

    FILE *fichier = fopen(path, "r");
    if (fichier == NULL) {
        printf("Erreur lors du chargement\n");
        return NULL;
    }

    char ligne[1024];

    int dim = 0;
    int n = 0;

    while (fgets(ligne, sizeof(ligne), fichier) != NULL) {
        n++;
        dim=0;
        ligne[strcspn(ligne, "\r\n")] = 0;

        char *line = strtok(ligne, ",");
        while (line != NULL) {
            dim++;
            line = strtok(NULL, ",");
        }
    }

    fclose(fichier);

    // Cree le dataset
    Dataset *d = new_dataset(dim,n);

    // Remplissage

    fichier = fopen(path, "r");
    if (fichier == NULL) {
        printf("Erreur lors du chargement\n");
        return NULL;
    }
    
    int i = 0;
    while (fgets(ligne, sizeof(ligne), fichier) != NULL) {
        ligne[strcspn(ligne, "\r\n")] = 0;
        int j = 0;

        char *line = strtok(ligne, ",");
        while (line != NULL) {
            
            d->data[i][j] = strtof(line,NULL);

            line = strtok(NULL, ",");
            j++;
        }
        i++;
    }

    fclose(fichier);

    

    return d;
};

/**
 * \fn int sigmoide(int z)
 */ 
float sigmoide(float z){
    return (1 / (1 + exp(-z)));
}


/**
 * \fn void shuffle(Dataset *d)
 * \brief Mélange les données d'un dataset
 *
 * \param d dataset à mélanger
 */ 
void shuffle(Dataset *d){

    int n = d->n;

    for(int i=0; i<d->n; i++){
        int j = rand() % n;
        float *temp = d->data[i];
        d->data[i] = d->data[j];
        d->data[j] = temp;
    }

}



/**
 * \fn void normalize(Dataset *d)
 * \brief Normalise les données d'un dataset
 *
 * \param d dataset à normaliser
 */ 
void normalize(Dataset *d){
    
    int n = d->n;
    int dim = d->dim;

    for(int j=0; j<dim-1; j++){
        float min = d->data[0][j];
        float max = d->data[0][j];

        for(int i=1; i<n; i++){
            if(d->data[i][j] < min){
                min = d->data[i][j];
            }
            if(d->data[i][j] > max){
                max = d->data[i][j];
            }
        }

        for(int i=0; i<n; i++){
            d->data[i][j] = (d->data[i][j] - min) / (max - min);
        }
    }
}


/**
 * \fn void split_dataset(Dataset *d, Dataset *train, Dataset *test, float ratio)
 * \brief Sépare les données en deux dataset
 *
 * \param ratio 0 -> 1,EX 0.8 = 80%
 */ 
void split_dataset(Dataset *d, Dataset **train, Dataset **test, float ratio){
    int dim = d->dim;
    int n = d->n;
    int nTrain = n*ratio;
    int nTest = n - nTrain;
    
    Dataset *resTrain = new_dataset(dim,nTrain);
    Dataset *resTest = new_dataset(dim,nTest);
    for(int i = 0; i<n;i++){
        if(i<nTrain){
            for(int j= 0; j<dim; j++){
                resTrain->data[i][j]=d->data[i][j];
            }
        }else{
            for(int j= 0; j<dim; j++){
                resTest->data[i-nTrain][j]=d->data[i][j];
            }
        }
    }
    *train=resTrain;
    *test=resTest;
    free_dataset(d);
}