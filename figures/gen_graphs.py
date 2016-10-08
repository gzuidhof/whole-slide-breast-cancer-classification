import pandas as pd
from glob import glob
import os
import matplotlib.pyplot as plt
plt.style.use('ggplot')

max_epoch=120
models_folder = '../models/'
model_instances = sorted(glob(models_folder+'*'), reverse=True)

def get_metrics(model_name):
    for m in model_instances:
        if model_name in m:
            print "Model ", model_name, "using folder", m
            return pd.DataFrame.from_csv(os.path.join(models_folder, m, 'metrics.csv'))


names2=['resnet_2class', 'vggnet_2class']
labels2=["WRN-4-2", "VGG-16"]
colors2=['b','g']


names3=['resnet_3class', 'vggnet_3class']
labels3=["WRN-4-2", "VGG-16"]
colors3=['b','g']


best = []


def make_plot(names, labels, colors, title, out_file, max_epoch):

    dfs = map(get_metrics, names)

    plt.figure()
    #plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim((0.5,1))

    handles = []

    for df, l, c in zip(dfs, labels, colors):
        acc_val = df['Accuracy_val'].values[:max_epoch]
        best.append(max(acc_val))
        acc_train = df['Accuracy_train'].values[:max_epoch]

        handles.append ( plt.plot(acc_val, label=l, color=c)[0] )
        plt.plot(acc_train, label=l, color=c, linestyle='dashed')

    plt.legend(handles=handles, loc=2)
    plt.savefig(out_file,bbox_inches='tight')
    #plt.show()

make_plot(names3, labels3, colors3, title="3 class CNN performance", out_file='output/3class_performance.png', max_epoch=max_epoch)
make_plot(names2, labels2, colors2, title="2 class CNN performance", out_file='output/2class_performance.png', max_epoch=max_epoch)

names4=['stack_on_2class_768', 'stack_on_3class_768']
labels4=["Stacked on 2 class", "Stacked on 3 class"]
colors4=['b','g']

names5=['stack_on_3class_512', 'stack_on_3class_768', 'stack_on_3class_1024']
labels5=["512x512 input", "768x768 input", "1024x1024 input"]
colors5=['b', 'g', 'm']

make_plot(names4, labels4, colors4, title="", out_file='output/768_performance.png', max_epoch=120)
make_plot(names5, labels5, colors5, title="", out_file='output/patch_sizes_performance.png', max_epoch=120)

best = ['{0:.4f}'.format(b) for b in best]

print best

table224 = r"""
\begin{table}[!t]
%% increase table row spacing, adjust to taste
\renewcommand{\arraystretch}{1.1}
% if using array.sty, it might be a good idea to tweak the value of
% \extrarowheight as needed to properly center the text within the cells
\caption{Best epoch patch-level accuracy of 224x224 networks}
\label{table_results_224}
\centering
%% Some packages, such as MDW tools, offer better commands for making tables
%% than the plain LaTeX2e tabular which is used here.
\begin{tabular}{|llr|}
\hline
\textsc{Labels}&\textsc{Architecture}&\textsc{Accuracy}\\
\hline
\textit{Benign, Cancer}&&\\
&VGG-16&  __ACCVGG2__\\
&WRN-4-2& __ACCWR2__\\
\hline
\textit{Benign, DCIS, IDC}&&\\
&VGG-16& __ACCVGG3__\\
&WRN-4-2& __ACCWR3__\\
\hline
\end{tabular}
\end{table}

""".replace("__ACCWR2__", best[2]).replace('__ACCVGG2__', best[3]).replace('__ACCWR3__', best[0]).replace('__ACCVGG3__', best[1])

#print a % 123, 123, 123, 123, 123, 123, 123

with open('output/table_accuracy_224.tex', 'w') as f:
    f.write(table224)



table_stacked = r"""
\begin{table}[!t]
%% increase table row spacing, adjust to taste
\renewcommand{\arraystretch}{1.1}
% if using array.sty, it might be a good idea to tweak the value of
% \extrarowheight as needed to properly center the text within the cells
\caption{Best epoch patch-level accuracy of stacked networks}
\label{table_results_stacked}
\centering
%% Some packages, such as MDW tools, offer better commands for making tables
%% than the plain LaTeX2e tabular which is used here.
\begin{tabular}{|llr|}
\hline
\textsc{Input patch size}\pbox{2cm}{}&\textsc{Stacked on}&\textsc{Accuracy}\\
\hline
\textit{512x512}&&\\
&3 Class WRN-4-2& 512PERFORMANCE \\
%\hline
\textit{768x768}&&\\
&2 Class WRN-4-2&  768PERFORMANCE_2 \\
&3 Class WRN-4-2& 768PERFORMANCE_3\\
%\hline
\textit{1024x1024}&&\\
&3 Class WRN-4-2& 1024PERFORMANCE\\
\hline

\end{tabular}
\end{table}


""".replace('512PERFORMANCE',best[6]).replace('768PERFORMANCE_2', best[4]).replace(
    '768PERFORMANCE_3', best[5]).replace('1024PERFORMANCE', best[8])

with open('output/table_accuracy_stacked.tex', 'w') as f:
    f.write(table_stacked)


#metrics_files = 
