import numpy as np
import matplotlib.pyplot as plt

import prepostprocessing.pre_processing as preproc

QAPF_upper_regions_numbers = {
    np.nan: 0,
    'quartzolite': 1,
    'quartz-rich granitoid': 2,
    'alkali feldspar granite': 3,
    'syeno granite': 4,
    'monzo granite': 5,
    'granodiorite': 6,
    'tonalite': 7,
    'quartz alkali\nfeldspar syenite': 8,
    'quartz syenite': 9,
    'quartz monzonite': 10,
    'quartz monzodiorite\nquartz monzogabbro': 11,
    'quartz diorite\nquartz gabbro\nquartz anorthosite': 12,
    'alkali feldspar syenite': 13,
    'syenite': 14,
    'monzonite': 15,
    'monzodiorite monzogabbro': 16,
    'diorite gabbro anorthosite': 17,
    }

QAPF_upper_regions_colors = {}
for number, (key, value) in zip(range(0, 20, 1),
                                QAPF_upper_regions_numbers.items()):
    QAPF_upper_regions_colors[int(value)] = number/20


def discrete_cmap(N, base_cmap=None):
    """https://gist.github.com/jakevdp/91077b0cae40f8f8244a"""
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return plt.cm.get_cmap(base_cmap, N)
    # return base.from_list(cmap_name, color_list, N)

# QAPF_upper_regions_colors = {
#     np.nan: 'white',
#     'quartzolite': 'black',
#     'quartz-rich granitoid': 'black',
#     'alkali feldspar granite': 3,
#     'syeno granite': 4,
#     'monzo granite': 5,
#     'granodiorite': 6,
#     'tonalite': 7,
#     'quartz alkali\nfeldspar syenite': 8,
#     'quartz syenite': 9,
#     'quartz monzonite': 10,
#     'quartz monzodiorite\nquartz monzogabbro': 11,
#     'quartz diorite\nquartz gabbro\nquartz anorthosite': 12,
#     'alkali feldspar syenite': 13,
#     'syenite': 14,
#     'monzonite': 15,
#     'monzodiorite monzogabbro': 16,
#     'diorite gabbro anorthosite':17,
#     }

# Upper row = Quartz
# Middle row = Plagioclase
# Lower row  = Alkalifeldspar
QAPF_upper_regions = {
    "quartzolite":
    np.array([[ 100., 100.,  90.,  90.],
              [   0., 100., 100.,   0.],
              [ 100.,   0.,   0., 100.]]),

    "quartz-rich granitoid":
    np.array([[ 90.,  90.,  60.,  60.],
              [  0., 100., 100.,   0.],
              [100.,   0.,   0., 100.]]),

    "alkali feldspar granite":
    np.array([[ 60.,  60., 20.,  20.],
              [  0.,  10., 10.,   0.],
              [100.,  90., 90., 100.]]),

    "syeno granite":
    np.array([[ 60., 60., 20., 20.],
              [ 10., 35., 35., 10.],
              [ 90., 65., 65., 90.]]),

    "monzo granite":
    np.array([[ 60., 60., 20., 20.],
              [ 35., 65., 65., 35.],
              [ 65., 35., 35., 65.]]),

    "granodiorite":
    np.array([[ 60., 60., 20., 20.],
              [ 65., 90., 90., 65.],
              [ 35., 10., 10., 35.]]),

    "tonalite":
    np.array([[ 60.,  60.,  20., 20.],
              [ 90., 100., 100., 90.],
              [ 10.,   0.,   0., 10.]]),

    "quartz alkali\nfeldspar syenite":
    np.array([[ 20.,  20.,   5.,  5.],
              [  0.,  10., 10.,   0.],
              [100.,  90., 90., 100.]]),

    "quartz syenite":
    np.array([[ 20., 20.,  5.,  5.],
              [ 10., 35., 35., 10.],
              [ 90., 65., 65., 90.]]),

    "quartz monzonite":
    np.array([[ 20., 20.,  5.,  5.],
              [ 35., 65., 65., 35.],
              [ 65., 35., 35., 65.]]),

    "quartz monzodiorite\nquartz monzogabbro":
    np.array([[ 20., 20.,  5.,  5.],
              [ 65., 90., 90., 65.],
              [ 35., 10., 10., 35.]]),

    "quartz diorite\nquartz gabbro\nquartz anorthosite":
    np.array([[ 20.,  20.,   5.,  5.],
              [ 90., 100., 100., 90.],
              [ 10.,   0.,   0., 10.]]),

    "alkali feldspar syenite":
    np.array([[  5.,   5.,  0.,   0.],
              [  0.,  10., 10.,   0.],
              [100.,  90., 90., 100.]]),

    "syenite":
    np.array([[  5.,  5.,  0.,  0.],
              [ 10., 35., 35., 10.],
              [ 90., 65., 65., 90.]]),

    "monzonite":
    np.array([[  5.,  5.,  0.,  0.],
              [ 35., 65., 65., 35.],
              [ 65., 35., 35., 65.]]),

    "monzodiorite monzogabbro":
    np.array([[  5.,  5.,  0.,  0.],
              [ 65., 90., 90., 65.],
              [ 35., 10., 10., 35.]]),

    "diorite gabbro anorthosite":
    np.array([[  5.,   5.,  0.,   0.],
              [ 90., 100., 100., 90.],
              [ 10.,   0.,   0., 10.]]),
       }


def prepare_df_for_qapf_classification(df):
    df_copy = df.copy()
    try:
        df_copy.columns = ["Q", "P", "A", "R"]
    except ValueError:
        df_copy = df_copy.rename(columns={"K": "A"})
    df_copy = df_copy.loc[:, ["Q", "P", "A"]]
    df_copy = preproc.normalize(df_copy).values
    return df_copy


def check_QAPF_region(df, regions=QAPF_upper_regions, return_points=False):
    """Classify normalized QAPF compositions according to
    the QAPF (or Streckeisen) classification diagram"""

    points = prepare_df_for_qapf_classification(df)

    classification = []

    for i, point in enumerate(points):
        if all(np.isnan(point)):
            classification.append(np.nan)
        elif not np.isclose(point.sum(), 100.0):
            raise ValueError(f"sum of components is not equal to 100.0 for point {i}.")
        else:
            for name, region in regions.items():

                point_1_ = point[1]/(point[1]+point[2]) * 100
                point_2_ = point[2]/(point[1]+point[2]) * 100

                if ((region[0].min() <= point[0] <= region[0].max()) &
                   (region[1].min() <= point_1_ <= region[1].max()) &
                   (region[2].min() <= point_2_ <= region[2].max())):

                    classification.append(name)
                    break

    if return_points:
        return classification, points
    else:
        return classification


# def plot_contour_map_interpolated_QAPF(interpolation_array,
#                                        grid,
#                                        coordinates_utm_qapf,
#                                        pluton,
#                                        values_to_plot,
#                                        number_of_classes=21,
#                                        skip_xaxis_label=0,
#                                        skip_yaxis_label=0,
#                                        skip_xaxis_start=0,
#                                        skip_yaxis_start=0,
#                                        legend_outside_plot=False,
#                                        multiplier=0.75,
#                                        bbox_x_anchor=1.3,
#                                        plot_control_points=True,
#                                        show_qapf_control_points=True,
#                                        marker_size=1,
#                                        **kwargs):
#     fig, ax = plt.subplots(**kwargs)

#     cmap_custom, norm_custom = create_custom_colormap()

#     levels_custom = [i for i in range(number_of_classes + 1)]

#     contour = ax.contourf(grid[0], grid[1], interpolation_array,
#                           cmap=cmap_custom, norm=norm_custom,
#                           levels=levels_custom)

#     if plot_control_points:
#         if show_qapf_control_points:
#             for index, row in coordinates_utm_qapf.iterrows():
#                 ax.plot(row["X"],
#                         row["Y"],
#                         color=cmap_custom(QAPF_upper_regions_numbers_vs_cmap[QAPF_upper_regions_numbers[row["QAPF"]] - 1]),
#                         marker='o',
#                         markeredgecolor='k',
#                         markeredgewidth=0.5,
#                         ms=2,
#                         linestyle='None')
#         else:
#             ax.plot(coordinates_utm_qapf["X"],
#                     coordinates_utm_qapf["Y"],
#                     color='k',
#                     marker='o',
#                     ms=marker_size,
#                     linestyle='None')
#     # plt.colorbar(contour)

#     # proxy = [plt.Rectangle((0, 0), 1, 1,
#     #          fc=cmap_custom(QAPF_upper_regions_numbers_vs_cmap[value]))
#     #          for value in values_to_plot]

#     proxy = [plt.Rectangle((0, 0), 1, 1, fc=pc.get_facecolor()[0])
#              for pc in contour.collections]

#     values = [level for level in contour.levels if level in values_to_plot]
#     values = [value - 1 for value in map(int, values)]

#     # print(len(proxy))
#     proxy_items = [proxy[value - 1] for value in values_to_plot]
#     labels = [QAPF_upper_regions_numbers_inverse[value + 1]
#               for value in values]
#     # print(values_to_plot)
#     # labels = [QAPF_upper_regions_numbers_inverse[value]
#     #           for value in values_to_plot]

#     # plt.legend(proxy_items, labels)

#     if skip_xaxis_label != 0:
#         every_nth = skip_xaxis_label
#         for n, label in enumerate(ax.xaxis.get_ticklabels(),
#                                   start=skip_xaxis_start):
#             if n % every_nth != 0:
#                 label.set_visible(False)

#     if skip_yaxis_label != 0:
#         every_nth = skip_yaxis_label
#         for n, label in enumerate(ax.yaxis.get_ticklabels(),
#                                   start=skip_yaxis_start):
#             if n % every_nth != 0:
#                 label.set_visible(False)

#     # Set the color of the visible spines
#     for spine in ['left', 'right', 'top', 'bottom']:
#         ax.spines[spine].set_color('grey')

#     # Set general tick parameters
#     ax.tick_params(axis='both',
#                    direction='in',
#                    colors='grey',
#                    labelsize=9)

#     ax.set_aspect('equal', adjustable='box')

#     fig.patch.set_facecolor('white')

#     if legend_outside_plot:
#         # Shrink current axis by 20%
#         box = ax.get_position()
#         ax.set_position([box.x0, box.y0, box.width * multiplier, box.height])

#         # # Put a legend below current axis
#         ax.legend(proxy_items, labels, loc='upper center',
#                   bbox_to_anchor=(bbox_x_anchor, 1),
#                   fancybox=True, shadow=True, ncol=1, title=None,
#                   fontsize='x-small')
#     else:
#         ax.legend(proxy_items, labels, ncol=1, title="", prop={'size': 8})
#         plt.tight_layout()
#     if show_qapf_control_points:
#         plt.savefig(f"../_FIGURES/qapf_contour/{pluton}_QAPF_interpolated_control.pdf")
#     else:
#         plt.savefig(f"../_FIGURES/qapf_contour/{pluton}_QAPF_interpolated.pdf")

#     plt.show()
