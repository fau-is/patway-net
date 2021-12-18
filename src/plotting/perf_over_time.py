import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


cut_lengths = range(1, 21)

###### Accuracy Values
#mean_line_stat_ACC = np.array([0.908040201005024, 0.908040201005024, 0.908040201005024, 0.903684210526315, 0.903684210526315, 0.895428571428571, 0.889090909090909, 0.888414634146341, 0.879729729729729, 0.874285714285714, 0.865116279069767, 0.852542372881356, 0.831372549019607, 0.790243902439024, 0.750724637681159, 0.708771929824561, 0.622727272727272, 0.611428571428571, 0.593548387096774, 0.553846153846153])
#mean_line_seq_ACC = np.array([0.904522613065326, 0.904522613065326, 0.904522613065326, 0.898947368421053, 0.90157894736842, 0.894285714285714, 0.893333333333333, 0.892073170731707, 0.949324324324323, 0.976428571428571, 0.990697674418604, 0.989830508474576, 0.988235294117647, 0.985365853658536, 0.982608695652173, 0.980701754385965, 0.995454545454545, 0.991428571428571, 0.990322580645161, 0.992307692307692])
mean_line_RF_ACC = np.array([0.904522613065326, 0.904522613065326, 0.904522613065326, 0.9, 0.9, 0.891428571428571, 0.884848484848484, 0.884146341463414, 0.871621621621621, 0.864285714285714, 0.852713178294573, 0.838983050847457, 0.813725490196078, 0.768292682926829, 0.72463768115942, 0.666666666666666, 0.570454545454545, 0.548571428571428, 0.532258064516128, 0.488461538461538])
mean_line_NB_ACC = np.array([0.904522613065326, 0.904522613065326, 0.904522613065326, 0.9, 0.9, 0.891428571428571, 0.884848484848484, 0.884146341463414, 0.871621621621621, 0.864285714285714, 0.852713178294573, 0.838983050847457, 0.813725490196078, 0.768292682926829, 0.72463768115942, 0.649122807017543, 0.545454545454545, 0.542857142857142, 0.516129032258064, 0.5])
mean_line_LR_ACC = np.array([0.914572864321608, 0.909547738693467, 0.914572864321608, 0.91578947368421, 0.910526315789473, 0.88, 0.872727272727272, 0.878048780487804, 0.945945945945945, 0.971428571428571, 0.984496124031007, 0.97457627118644, 0.96078431372549, 0.939024390243902, 0.942028985507246, 0.912280701754385, 0.886363636363636, 0.857142857142857, 0.838709677419354, 0.846153846153846])
mean_line_KNN_ACC = np.array([0.914572864321608, 0.914572864321608, 0.894472361809045, 0.878947368421052, 0.873684210526315, 0.845714285714285, 0.842424242424242, 0.841463414634146, 0.824324324324324, 0.807142857142857, 0.75968992248062, 0.73728813559322, 0.725490196078431, 0.658536585365853, 0.623188405797101, 0.596491228070175, 0.522727272727272, 0.485714285714285, 0.483870967741935, 0.5, ])
mean_line_GB_ACC = np.array([0.944723618090452, 0.944723618090452, 0.944723618090452, 0.936842105263157, 0.936842105263157, 0.925714285714285, 0.921212121212121, 0.920731707317073, 0.945945945945945, 0.964285714285714, 0.968992248062015, 0.966101694915254, 0.96078431372549, 0.951219512195121, 0.956521739130434, 0.947368421052631, 0.931818181818181, 0.939999999999999, 0.932258064516128, 0.919230769230769])
#mean_line_DT_ACC = np.array([0.924623115577889, 0.924623115577889, 0.924623115577889, 0.91578947368421, 0.901578947368421, 0.898857142857142, 0.892727272727272, 0.898170731707317, 0.939189189189189, 0.957142857142857, 0.968992248062015, 0.966101694915254, 0.96078431372549, 0.963414634146341, 0.947826086956521, 0.936842105263157, 0.918181818181818, 0.897142857142856, 0.883870967741935, 0.861538461538461])
mean_line_ADA_ACC = np.array([0.904522613065326, 0.904522613065326, 0.904522613065326, 0.9, 0.9, 0.897142857142857, 0.89090909090909, 0.890243902439024, 0.945945945945945, 0.971428571428571, 0.984496124031007, 0.983050847457627, 0.980392156862745, 0.975609756097561, 0.971014492753623, 0.964912280701754, 0.954545454545454, 0.942857142857142, 0.967741935483871, 0.961538461538461])
mean_line_complete_ACC = np.array([0.906030150753769, 0.907035175879397, 0.906030150753768, 0.898421052631578, 0.9, 0.896571428571428, 0.893939393939394, 0.892682926829268, 0.94054054054054, 0.967857142857142, 0.982945736434108, 0.977966101694915, 0.976470588235294, 0.973170731707317, 0.972463768115941, 0.971929824561403, 0.984090909090909, 0.985714285714286, 0.980645161290322, 0.980769230769231])

max_line_complete_ACC = np.array([0.919143706885125, 0.919993765787163, 0.919334876678265, 0.912155777211262, 0.921314164980333, 0.921275390081391, 0.920667972207515, 0.917103639505488, 0.963313918305882, 0.990725443705117, 1.0018706288616, 1.00316282838528, 1.00016283524234, 0.999212385980564, 0.994395283986118, 0.988009037526161, 1.00184147653615, 1.00488058266428, 1.00204274058294, 1])
min_line_complete_ACC = np.array([0.892916594622412, 0.89407658597163, 0.892725424829272, 0.884686328051895, 0.878685835019667, 0.871867467061465, 0.867210815671273, 0.868262214153047, 0.917767162775199, 0.944988842009168, 0.964020844006614, 0.952769375004545, 0.952778341228245, 0.947129077434069, 0.950532252245765, 0.955850611596646, 0.966340341645666, 0.966547988764287, 0.959247581997707, 0.961538461538461])




###### AUC Values
#mean_line_stat_AUC = np.array([0.727280701754386, 0.727280701754386, 0.727280701754386, 0.732163742690058, 0.732163742690058, 0.733434547908232, 0.72606344628695, 0.724936479128856, 0.730436556507548, 0.731709438886472, 0.73267942583732, 0.731738437001594, 0.732593532022828, 0.704427736006683, 0.704842105263158, 0.717451523545706, 0.710526315789473, 0.681907894736842, 0.6375, 0.669047619047618])
#mean_line_seq_AUC = np.array([0.5, 0.530994152046783, 0.467383040935672, 0.357633117882425, 0.491012619267466, 0.610863697705803, 0.645962509012256, 0.676188747731397, 0.840146878824969, 0.882340147890387, 0.963492822966507, 0.960818713450292, 0.963601775523145, 0.964745196324143, 0.967368421052631, 0.963988919667589, 0.996842105263158, 0.996052631578947, 0.995416666666667, 0.995238095238095])
mean_line_RF_AUC = np.array([0.776447368421052, 0.771783625730994, 0.778801169590643, 0.721945213911973, 0.734995383194829, 0.734547908232118, 0.723035328046142, 0.730852994555354, 0.865891472868217, 0.874032187907786, 0.90956937799043, 0.910207336523126, 0.914965123652504, 0.934001670843776, 0.94, 0.934764542936288, 0.914947368421052, 0.899999999999999, 0.895416666666666, 0.901785714285714])
mean_line_NB_AUC = np.array([0.683187134502923, 0.675292397660818, 0.680847953216374, 0.394275161588181, 0.370883348722683, 0.32051282051282, 0.325522710886806, 0.32087114337568, 0.314973480212158, 0.299260548064375, 0.232535885167464, 0.31578947368421, 0.447685478757133, 0.467000835421888, 0.536842105263157, 0.581717451523545, 0.534736842105263, 0.56578947368421, 0.554166666666666, 0.547619047619047])
mean_line_LR_AUC = np.array([0.692251461988304, 0.677339181286549, 0.689619883040935, 0.610649430594028, 0.63496460449369, 0.671052631578947, 0.673035328046142, 0.680580762250453, 0.847001223990208, 0.886472379295345, 0.927272727272727, 0.931419457735247, 0.927076727964489, 0.927318295739348, 0.919999999999999, 0.9196675900277, 0.913684210526315, 0.891447368421052, 0.8875, 0.892857142857142, ])
mean_line_KNN_AUC = np.array([0.707163742690058, 0.689473684210526, 0.584356725146198, 0.520775623268698, 0.491381963681132, 0.486336032388663, 0.528658976207642, 0.524319419237749, 0.523051815585475, 0.461722488038277, 0.45334928229665, 0.459861775651249, 0.511414077362079, 0.518379281537176, 0.487894736842105, 0.524238227146814, 0.494736842105263, 0.5, 0.51875, 0.571428571428571])
mean_line_GB_AUC = np.array([0.837894736842105, 0.837894736842105, 0.837309941520467, 0.80932594644506, 0.815851031086487, 0.803778677462888, 0.799711607786589, 0.800725952813066, 0.867809057527539, 0.869856459330143, 0.926794258373205, 0.927166400850611, 0.928979074191502, 0.940685045948203, 0.941052631578947, 0.939058171745152, 0.936842105263157, 0.921052631578947, 0.9125, 0.904761904761904])
#mean_line_DT_AUC = np.array([0.708479532163742, 0.708479532163742, 0.708479532163742, 0.710987996306555, 0.703185595567867, 0.717780026990553, 0.717555875991348, 0.721234119782214, 0.870053039575683, 0.911918225315354, 0.937320574162679, 0.93806485911749, 0.94007609384908, 0.946115288220551, 0.914421052631578, 0.910110803324099, 0.903999999999999, 0.883223684210526, 0.872083333333333, 0.841666666666666])
mean_line_ADA_AUC = np.array([0.854239766081871, 0.854239766081871, 0.862280701754386, 0.845029239766081, 0.841797476146506, 0.809885290148448, 0.804434030281182, 0.804537205081669, 0.876172990616075, 0.889082209656372, 0.936363636363636, 0.936204146730462, 0.937856689917565, 0.978279030910609, 0.978947368421052, 0.976454293628808, 0.972631578947368, 0.957236842105263, 0.954166666666666, 0.946428571428571])
mean_line_complete_AUC = np.array([0.699239766081871, 0.697748538011695, 0.687309941520467, 0.613142505386272, 0.675777162203755, 0.72246963562753, 0.719322278298485, 0.730998185117967, 0.85593635250918, 0.890474119182253, 0.965167464114832, 0.966241360978203, 0.968421052631579, 0.965914786967418, 0.965473684210526, 0.966620498614958, 0.999578947368421, 0.999671052631579, 0.999166666666667, 0.999404761904762])

max_line_complete_AUC = np.array([0.710294911376021, 0.71611744219833, 0.702296826340087, 0.65451346499129, 0.710649091021533, 0.757559041794808, 0.759083619354697, 0.76891817239418, 0.876209685279234, 0.911760908454566, 0.973693881934101, 0.974276747748189, 0.976258472824593, 0.972062947198988, 0.972547493733226, 0.972950911291981, 1.00042105263158, 1.00065789473684, 1.00083333333333, 1.00119047619048])
min_line_complete_AUC = np.array([0.68818462078772, 0.679379633825061, 0.672323056700847, 0.571771545781255, 0.640905233385976, 0.687380229460251, 0.679560937242274, 0.693078197841753, 0.835663019739125, 0.86918732990994, 0.956641046295564, 0.958205974208217, 0.960583632438564, 0.959766626735848, 0.958399874687827, 0.960290085937935, 0.998736842105263, 0.998684210526316, 0.9975, 0.997619047619048])


######

def plot_line_plots(cut_lengths, means_acc, mins_acc, maxes_acc, 
                    means_auc, mins_auc, maxes_auc, labels):
    palet = sns.color_palette("tab10")
    matplotlib.rcParams.update({'font.size': 24})
    
    fig, axs = plt.subplots(1, 2, figsize=(22, 10))
    ax1 = axs[0]
    ax2 = axs[1]
    
    def plot_on_axes(ax, m, a, b, title):
        for i, (mean, a, b, l) in enumerate(zip(m, a, b, labels)):
            if i == 0:
                ax.plot(cut_lengths, mean, color=palet[i], label=l)
            else:
                ax.plot(cut_lengths, mean, color=palet[i], linestyle='dashed', label=l)
            if len(a) > 0 and len(b) > 0:
                ax.fill_between(cut_lengths, a, b, alpha=.2, color=palet[i])
        # ax.set_title(r'$M_{%s}$' % target_activity_abbreviation, fontsize=30)
        # ax.set_xlabel('Size of Process Instance Prefix for Prediction')
        ax.set_xticks(np.arange(0, 20 + 1, step=5))
        ax.set_ylabel('Predictive performance')
        ax.set_ylim(0.45, 1.01)
        ax.title.set_text(title)
        
    plot_on_axes(ax1, means_acc, mins_acc, maxes_acc, title='Accuracy')
    plot_on_axes(ax2, means_auc, mins_auc, maxes_auc, title='AUC$_{ROC}$')
    
    ax1.legend(ncol=1, loc='lower left', 
           columnspacing=1.3, labelspacing=0.0,
           handletextpad=0.0, handlelength=1.5,
           fancybox=False, shadow=False)
    # ax2.set_yticks([])
    y_axis = ax2.axes.get_yaxis()
    y_axis.set_visible(False)

    fig.text(0.5, -0.01, 'Size of patient pathway prefix for prediction', ha='center')

    plt.tight_layout()
    plt.savefig('tmp.pdf', bbox_inches="tight")
    
    
# There are three args for acc and three args for auc
# Each arg is a list where the length equals the number of lines
# Each line is a list (or a 1D numpy array) with the y-values
# Pass an empty list in mins/maxes arg for a line not to have confidence intervals
plot_line_plots(cut_lengths, 
                means_acc = [mean_line_complete_ACC, mean_line_ADA_ACC, mean_line_GB_ACC, mean_line_RF_ACC, mean_line_LR_ACC], 
                mins_acc = [min_line_complete_ACC,[], [], [], []], 
                maxes_acc = [max_line_complete_ACC,[], [], [], []],
                means_auc = [mean_line_complete_AUC, mean_line_ADA_AUC, mean_line_GB_AUC, mean_line_RF_AUC, mean_line_LR_AUC],
                mins_auc = [min_line_complete_AUC,[], [], [], []], 
                maxes_auc = [max_line_complete_AUC, [],[], [], []],
                labels=['PatWay-Net $\pm$ SD', 'AdaBoost', 'Gradient Boosting','Random Forest',  'Logistic Regression'])