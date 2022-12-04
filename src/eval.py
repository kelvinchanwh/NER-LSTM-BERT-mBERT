from sklearn.metrics import classification_report

class Evaluate():
    def __init__(self):
        self.wnut_report = []
        self.surf_report = []

    def wnut_evaluate(self, txt, verbose = 0):
        '''entity evaluation: we evaluate by whole named entities'''
        npred = 0; ngold = 0; tp = 0
        nrows = len(txt)
        for i in txt.index:
            if txt['prediction'][i]=='B' and txt['bio_only'][i]=='B':
                npred += 1
                ngold += 1
            for predfindbo in range((i+1),nrows):
                if txt['prediction'][predfindbo]=='O' or txt['prediction'][predfindbo]=='B':
                    break  # find index of first O (end of entity) or B (new entity)
            for goldfindbo in range((i+1),nrows):
                if txt['bio_only'][goldfindbo]=='O' or txt['bio_only'][goldfindbo]=='B':
                    break  # find index of first O (end of entity) or B (new entity)
            if predfindbo==goldfindbo:  # only count a true positive if the whole entity phrase matches
                tp += 1
            elif txt['prediction'][i]=='B':
                npred += 1
            elif txt['bio_only'][i]=='B':
                ngold += 1
    
        fp = npred - tp  # n false predictions
        fn = ngold - tp  # n missing gold entities
        prec = tp / (tp+fp)
        rec = tp / (tp+fn)
        f1 = (2*(prec*rec)) / (prec+rec)
        if verbose:
            print('Sum of TP and FP = %i' % (tp+fp))
            print('Sum of TP and FN = %i' % (tp+fn))
            print('True positives = %i, False positives = %i, False negatives = %i' % (tp, fp, fn))
            print('Precision = %.4f, Recall = %.4f, F1 = %.4f' % (prec, rec, f1))
        return [prec, rec, f1]

    def evaluate(self, df_long, verbose = 0):
        if verbose:
            print (classification_report(df_long['bio_only'], df_long['prediction'], labels=["B", "I"], digits = 4))
        self.surf_report.append(classification_report(df_long['bio_only'], df_long['prediction'], labels=["B", "I"], digits = 4, output_dict = True))
        self.wnut_report.append(self.wnut_evaluate(df_long, verbose = verbose))

    def print_results(self):
        print ("Whole Entity")
        print ("Precision: %.2f"%((sum([report[0] for report in self.wnut_report])/len(self.wnut_report))*100.))
        print ("Recall: %.2f"%((sum([report[1] for report in self.wnut_report])/len(self.wnut_report))*100.))
        print ("F1: %.2f"%((sum([report[2] for report in self.wnut_report])/len(self.wnut_report))*100.))

        print ("\nSurface Level")
        print ("Precision: %.2f"%((sum([list(report["Weighted"].values())[0] for report in self.surf_report])/len(self.surf_report))*100.))
        print ("Recall: %.2f"%((sum([list(report["Weighted"].values())[1] for report in self.surf_report])/len(self.surf_report))*100.))
        print ("F1: %.2f"%((sum([list(report["Weighted"].values())[2] for report in self.surf_report])/len(self.surf_report))*100.))