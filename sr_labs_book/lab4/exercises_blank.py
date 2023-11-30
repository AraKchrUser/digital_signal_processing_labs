# Exercises in order to perform laboratory work


# Import of modules
import numpy as np
from matplotlib.pyplot import hist, plot, show, grid, title, xlabel, ylabel, legend, axis, imshow
from numba import njit, prange
from functools import partial


def tar_imp_hists(all_scores, all_labels):
    # Function to compute target and impostor histogram
    
    tar_scores = []
    imp_scores = []

    ###########################################################
    # Here is your code
    for i, label in enumerate(all_labels):
        if label:
            tar_scores.append(all_scores[i])
        else:
            imp_scores.append(all_scores[i])
    
    ###########################################################
    
    tar_scores = np.array(tar_scores)
    imp_scores = np.array(imp_scores)
    
    return tar_scores, imp_scores

def llr(all_scores, all_labels, tar_scores, imp_scores, gauss_pdf):
    # Function to compute log-likelihood ratio
    
    tar_scores_mean = np.mean(tar_scores)
    tar_scores_std  = np.std(tar_scores)
    imp_scores_mean = np.mean(imp_scores)
    imp_scores_std  = np.std(imp_scores)
    
    all_scores_sort   = np.zeros(len(all_scores))
    ground_truth_sort = np.zeros(len(all_scores), dtype='bool')
    
    ###########################################################
    # Here is your code
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    sorted_indexes = np.argsort(all_scores)
    all_scores_sort = all_scores[sorted_indexes]
    ground_truth_sort = all_labels[sorted_indexes].astype(bool)
    
    ###########################################################
    
    tar_gauss_pdf = np.zeros(len(all_scores))
    imp_gauss_pdf = np.zeros(len(all_scores))
    LLR           = np.zeros(len(all_scores))
    
    ###########################################################
    # Here is your code
    tar_gauss_pdf = gauss_pdf(all_scores_sort, tar_scores_mean, tar_scores_std)
    imp_gauss_pdf = gauss_pdf(all_scores_sort, imp_scores_mean, imp_scores_std)
    LLR = np.log(tar_gauss_pdf/imp_gauss_pdf)
    
    ###########################################################
    
    return ground_truth_sort, all_scores_sort, tar_gauss_pdf, imp_gauss_pdf, LLR

def map_test(ground_truth_sort, LLR, tar_scores, imp_scores, P_Htar):
    # Function to perform maximum a posteriori test
    
    len_thr = len(LLR)
    fnr_thr = np.zeros(len_thr)
    fpr_thr = np.zeros(len_thr)
    P_err   = np.zeros(len_thr)
    
    for idx in range(len_thr):
        solution = LLR > LLR[idx]                                      # decision
        
        err = (solution != ground_truth_sort)                          # error vector
        
        fnr_thr[idx] = np.sum(err[ ground_truth_sort])/len(tar_scores) # prob. of Type I  error P(Dimp|Htar), false negative rate (FNR)
        fpr_thr[idx] = np.sum(err[~ground_truth_sort])/len(imp_scores) # prob. of Type II error P(Dtar|Himp), false positive rate (FPR)
        
        P_err[idx]   = fnr_thr[idx]*P_Htar + fpr_thr[idx]*(1 - P_Htar) # prob. of error
    
    # Plot error's prob.
    plot(LLR, P_err, color='blue')
    xlabel('$LLR$'); ylabel('$P_e$'); title('Probability of error'); grid(); show()
        
    P_err_idx = np.argmin(P_err) # argmin of error's prob.
    P_err_min = fnr_thr[P_err_idx]*P_Htar + fpr_thr[P_err_idx]*(1 - P_Htar)
    
    return LLR[P_err_idx], fnr_thr[P_err_idx], fpr_thr[P_err_idx], P_err_min

def neyman_pearson_test(ground_truth_sort, LLR, tar_scores, imp_scores, fnr):
    # Function to perform Neyman-Pearson test
    
    thr   = 0.0
    fpr   = 0.0
    
    ###########################################################
    # Here is your code
    len_thr = len(LLR)
    fpr_thr = np.zeros(len_thr)
    fnr_thr = np.zeros(len_thr)

    for idx in range(len_thr):
        
        solution = LLR > LLR[idx]
        err = (solution != ground_truth_sort)
        
        fpr_thr[idx] = np.sum(err[~ground_truth_sort] / len(imp_scores))
        fnr_thr[idx] = np.sum(err[ ground_truth_sort])/len(tar_scores)

    err_idx = np.argmin(np.abs(fnr_thr - fnr))
    thr = LLR[err_idx]
    fpr = fpr_thr[err_idx]
    
    ###########################################################
    
    return thr, fpr

@njit(parallel=True)
def bayes_test(ground_truth_sort, LLR, tar_scores, imp_scores, P_Htar, C00, C10, C01, C11):
    # Function to perform Bayes' test
    
    thr   = 0.0
    fnr   = 0.0
    fpr   = 0.0
    AC    = 0.0
    
    ###########################################################
    # Here is your code
    # def calc_bayes(tpr, fpr, fnr, tnr, C00=C00, C10=C10, C01=C01, C11=C11, P_Htar=P_Htar):
    #     return C00 * tpr * P_Htar + C10 * fnr * P_Htar + C01 * fpr * (1 - P_Htar) + C11 * tnr * (1 - P_Htar)
    
    len_thr =  len(LLR)
    AC_err  = np.zeros(len_thr)
    fnr_thr = np.zeros(len_thr)
    fpr_thr = np.zeros(len_thr)
    tpr_thr = np.zeros(len_thr)
    tnr_thr = np.zeros(len_thr)
    
    # calc_bayes_ = partial(calc_bayes, C00=C00, C10=C10, C01=C01, C11=C11, P_Htar=P_Htar)
    
    for idx in prange(len_thr):
        
        solution = LLR > LLR[idx]
        err = (solution != ground_truth_sort)

        fnr_thr[idx] = np.sum(err [ ground_truth_sort])/len(tar_scores)
        fpr_thr[idx] = np.sum(err [~ground_truth_sort])/len(imp_scores)
        tpr_thr[idx] = np.sum(~err[ ground_truth_sort])/len(tar_scores)
        tnr_thr[idx] = np.sum(~err[~ground_truth_sort])/len(imp_scores)
        
        # AC_err[idx] = calc_bayes(tpr=tpr_thr[idx], fpr=fpr_thr[idx], fnr=fnr_thr[idx], tnr=tnr_thr[idx])
        tpr=tpr_thr[idx]
        fpr=fpr_thr[idx]
        fnr=fnr_thr[idx]
        tnr=tnr_thr[idx]
        AC_err[idx] = C00 * tpr * P_Htar + C10 * fnr * P_Htar + C01 * fpr * (1 - P_Htar) + C11 * tnr * (1 - P_Htar)
    
    AC_err_idx = np.argmin(AC_err)
    AC_err_min = AC_err.min()

    thr = LLR[AC_err_idx]
    fnr = fnr_thr[AC_err_idx]
    fpr = fpr_thr[AC_err_idx]
    AC  = AC_err_min

    # thr = 0.074
    # fnr = fnr_thr[0]
    # fpr = fpr_thr[0]
    # AC  = AC_err[0]
    

    ###########################################################
    return thr, fnr, fpr, AC

def minmax_test(ground_truth_sort, LLR, tar_scores, imp_scores, P_Htar_thr, C00, C10, C01, C11):
    # Function to perform minimax test
    
    thr    = 0.0
    fnr    = 0.0
    fpr    = 0.0
    AC     = 0.0
    P_Htar = 0.0
    
    ###########################################################
    # Here is your code
    from tqdm import tqdm
    from functools import partial
    
    results = []
    len_P_Htar_thr = len(P_Htar_thr)
    bayes_test_ = partial(
        bayes_test, 
        ground_truth_sort=ground_truth_sort, 
        LLR=LLR,
        tar_scores=tar_scores,
        imp_scores=imp_scores,
        C00 = C00,
        C10 = C10,
        C01 = C01,
        C11 = C11,
    )

    for idx in tqdm(range(len_P_Htar_thr)):
        p = P_Htar_thr[idx]
        thr, fnr, fpr, AC = bayes_test_(P_Htar=p)
        results.append({
            'thr': thr,
            'fnr': fnr,
            'fpr': fpr,
            'AC':  AC,
            'p':   p,
        })
    
    res    = max(results, key=lambda x: x["AC"])
    thr    = res["thr"]
    fnr    = res["fnr"]
    fpr    = res["fpr"]
    AC     = res["AC"]
    P_Htar = res["p"]
    ###########################################################
    
    return thr, fnr, fpr, AC, P_Htar, results