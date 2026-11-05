#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define LEN(a) ((int)(sizeof(a) / sizeof((a)[0])))

typedef struct {
    int epoques;
    float accBefore;
    float accAfter;
    int FN;
    int FP;
    int TN;
    int TP;
    int epoquesMax;
    float accMax;
    int valide;      // 1 si la ligne #END a bien été lue
    double duree;    // secondes
} Data;

typedef struct {
    const char *dataset;
    int epoques;
    float taux;
    float ratio;
    int normalized;
    int stopWhenTrained;
    int neurones;
    int couches;
    unsigned int seed;
} Config;

Data *getResults(const Config *c, int run, FILE *courbes){
    Data *res = calloc(1, sizeof(Data));
    char commande[512];
    char ligne[512];

    snprintf(commande, sizeof(commande),
             "./perceptron %s %d %f %f %d %d 0 %d %d %u",
             c->dataset, c->epoques, c->taux, c->ratio,
             c->normalized, c->stopWhenTrained,
             c->neurones, c->couches, c->seed);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    FILE *fp = popen(commande, "r");
    if (fp == NULL) {
        printf("Erreur lors de l'exécution de la commande.\n");
        return res;
    }

    while (fgets(ligne, sizeof(ligne), fp) != NULL) {

        if (strncmp(ligne, "#EP;", 4) == 0) {
            int ep, epMax, tp, fpos, tn, fn;
            float taux, acc, accMax;

            // #EP;epoque;taux;acc;accMax;epoqueMax;TP;FP;TN;FN
            if (sscanf(ligne, "#EP;%d;%f;%f;%f;%d;%d;%d;%d;%d",
                       &ep, &taux, &acc, &accMax, &epMax, &tp, &fpos, &tn, &fn) == 9) {
                fprintf(courbes, "%d,%s,%d,%.2f,%.4f,%d,%d,%u,%d,%.5f,%.4f,%.4f,%d,%d,%d,%d,%d\n",
                        run, c->dataset, c->normalized, c->ratio, c->taux,
                        c->neurones, c->couches, c->seed,
                        ep, taux, acc, accMax, epMax, tp, fpos, tn, fn);
            }

        } else if (strncmp(ligne, "#END;", 5) == 0) {
            // #END;epoques;accBefore;accAfter;FN;FP;TN;TP;epoquesMax;accMax
            int n = sscanf(ligne, "#END;%d;%f;%f;%d;%d;%d;%d;%d;%f",
                           &res->epoques, &res->accBefore, &res->accAfter,
                           &res->FN, &res->FP, &res->TN, &res->TP,
                           &res->epoquesMax, &res->accMax);
            res->valide = (n == 9);
        }
    }

    pclose(fp);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    res->duree = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;

    return res;
}

int main() {

    FILE *courbes = fopen("resultats_courbes.csv", "w");
    FILE *resume  = fopen("resultats_resume.csv", "w");
    if (courbes == NULL || resume == NULL) {
        perror("Erreur d'ouverture des fichiers CSV");
        return 1;
    }

    fprintf(courbes, "run,dataset,normalized,ratio,taux_initial,neurones,couches,seed,"
                     "epoque,taux,acc,accMax,epoqueMax,TP,FP,TN,FN\n");
    fprintf(resume,  "run,dataset,normalized,ratio,taux,neurones,couches,seed,"
                     "epoques_demandes,epoques_reels,accBefore,accAfter,FN,FP,TN,TP,"
                     "epoques_max,accMax,duree_s\n");

    const char *datasets[]    = {"./data/champignons.csv"};
    int         epoquesList[] = {5000};
    float       tauxList[]    = {0.05f, 0.1f, 0.3f};
    float       ratioList[]   = {0.8f};
    int         normList[]    = {1};
    int         neuronesList[]= {4,8,16, 32,64};
    int         couchesList[] = {1,2,3,4};
    unsigned    seeds[]       = {1,2,3,4,5};
    int         stopWhenTrained = 0;

    int run = 0;
    int total = LEN(datasets) * LEN(epoquesList) * LEN(tauxList) * LEN(ratioList)
              * LEN(normList) * LEN(neuronesList) * LEN(couchesList) * LEN(seeds);

    for (int a = 0; a < LEN(datasets); a++)
    for (int b = 0; b < LEN(epoquesList); b++)
    for (int c = 0; c < LEN(tauxList); c++)
    for (int d = 0; d < LEN(ratioList); d++)
    for (int e = 0; e < LEN(normList); e++)
    for (int f = 0; f < LEN(neuronesList); f++)
    for (int g = 0; g < LEN(couchesList); g++)
    for (int h = 0; h < LEN(seeds); h++) {

        Config cfg = {
            datasets[a], epoquesList[b], tauxList[c], ratioList[d], normList[e],
            stopWhenTrained, neuronesList[f], couchesList[g], seeds[h]
        };

        run++;
        Data *res = getResults(&cfg, run, courbes);
        if (res == NULL) continue;

        if (!res->valide) {
            printf("[%d/%d] Run sans ligne #END (crash ?) : taux=%.2f neurones=%d couches=%d seed=%u\n",
                   run, total, cfg.taux, cfg.neurones, cfg.couches, cfg.seed);
            free(res);
            continue;
        }

        printf("[%d/%d] taux=%.2f n=%d c=%d seed=%u | Ep: %d/%d | AccAfter: %.4f | AccMax: %.4f @%d | %.0fs\n",
               run, total, cfg.taux, cfg.neurones, cfg.couches, cfg.seed,
               res->epoques, cfg.epoques, res->accAfter, res->accMax, res->epoquesMax, res->duree);

        fprintf(resume, "%d,%s,%d,%.2f,%.4f,%d,%d,%u,%d,%d,%.4f,%.4f,%d,%d,%d,%d,%d,%.4f,%.1f\n",
                run, cfg.dataset, cfg.normalized, cfg.ratio, cfg.taux,
                cfg.neurones, cfg.couches, cfg.seed,
                cfg.epoques, res->epoques, res->accBefore, res->accAfter,
                res->FN, res->FP, res->TN, res->TP,
                res->epoquesMax, res->accMax, res->duree);

        fflush(courbes);
        fflush(resume);   // les CSV restent exploitables si tu interromps la grille
        free(res);
    }

    fclose(courbes);
    fclose(resume);
    return 0;
}