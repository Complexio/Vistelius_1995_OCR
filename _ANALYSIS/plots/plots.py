import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import ipywidgets as widgets
from ipywidgets import interact

from collections import defaultdict

from matplotlib.collections import PathCollection
from matplotlib.legend_handler import HandlerPathCollection

import numpy as np

from mpl_toolkits.axes_grid1 import make_axes_locatable

from nugget_estimation.kriging_tools import \
    calculate_and_sort_square_distance_matrix
from nugget_estimation.kriging_tools import find_points_within_search_radius

from matplotlib.colors import ListedColormap, BoundaryNorm

from qapf.qapf import QAPF_upper_regions_numbers


QAPF_upper_regions_numbers_inverse = {}
for (key, value), counter in zip(QAPF_upper_regions_numbers.items(),
                                 range(0, 18, 1)):
    QAPF_upper_regions_numbers_inverse[counter] = key

QAPF_upper_regions_numbers_vs_cmap = {}
for key, value in zip(range(0, 21, 1), np.arange(0, 1.05, 0.05)):
    QAPF_upper_regions_numbers_vs_cmap[key] = value

cmap_dict = defaultdict(lambda: mpl.cm.binary,
                        {"Q": mpl.cm.binary,
                         "P": mpl.cm.Reds,
                         "K": mpl.cm.Blues,
                         "Others": mpl.cm.Greens,
                         "B": mpl.cm.Greens,
                         "A": mpl.cm.Greens,
                         "H": mpl.cm.Greens,
                         })


def plot_cross_variogram(variogram_results, PCx, PCy, title=None,
                         save_abbrev=None, print_n_pairs=False,
                         search_radius=None):

    fig, ax = plt.subplots()

    lags = variogram_results[f"{PCx}{PCy}"][0]
    semivariance = variogram_results[f"{PCx}{PCy}"][1]
    n_pairs = variogram_results[f"{PCx}{PCy}"][2]

    plt.plot(lags, semivariance,
             color='k',
             marker='o',
             markersize=8,
             markeredgewidth=2,
             markeredgecolor="white",
             linewidth=3)

    if search_radius is not None:
        plt.vlines(search_radius,
                   min(semivariance),
                   max(semivariance)+0.00065,
                   color='k')

        plt.annotate(f"Search radius = {search_radius} m",
                     (search_radius, semivariance.mean()),
                     (search_radius * 1.1, semivariance.mean()))

    plt.title(title, color='grey', size=11)
    plt.xlabel("h", color='grey')
    plt.ylabel(f"{PCx}\n&\n{PCy}", color='grey', rotation=0, labelpad=20)
    plt.figtext(0.02, 0.85, r"$\gamma_{ij}(h)$", color='grey')

    # Change display of spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Set the color of the visible spines
    ax.spines['left'].set_color('grey')
    ax.spines['bottom'].set_color('grey')

    # Set the bounds of the spines
    ax.spines['bottom'].set_bounds(lags[0], lags[-1])
    ax.spines['left'].set_bounds(min(semivariance), max(semivariance)+0.00065)

    # Set general tick parameters
    ax.tick_params(axis='both',
                   direction='in',
                   colors='grey',
                   labelsize=9)

    # Set axes ticks
    ax.set_xticks(lags)
    # Set axes tick labels
    ax.set_xticklabels([f"{x:2.0f}" for x in lags],
                       fontsize=9,
                       rotation='horizontal')

    # Display number of pairs
    if print_n_pairs:
        for lag, semi, n in zip(lags, semivariance, n_pairs):
            plt.text(lag*1.01, semi, f"{n:4.0f}", color='grey')

    # Set facecolor of figure
    plt.gcf().set_facecolor('white')

    plt.tight_layout()
    if save_abbrev is not None:
        plt.savefig(
            f"../_FIGURES/cross_variogram_{save_abbrev}_{PCx}_{PCy}.pdf")
    plt.show()


def create_color_dict(data):
    multiplier = int((data.shape[0] / 10) + 1)
    colors = sns.color_palette() * multiplier

    color_dict = {}
    for i in range(0, data.shape[0]):
        color_dict[i] = colors[i]

    return color_dict


def annotate_plot(p, axer):
    x = p.get_bbox().get_points()[:, 0]
    y = int(p.get_bbox().get_points()[1, 1])
    axer.annotate(f"{y}",
                  (x.mean(), y),
                  size=12,
                  ha='center',
                  va='bottom')


def overview_plot_search_radius(coordinates, subset=["X", "Y"],
                                radius_min=300, radius_max=3100,
                                radius_step=100, n_cols=1):
    n_rows = int(((radius_max - radius_min) / radius_step) // n_cols)
    if (((radius_max - radius_min) / radius_step) % n_cols) != 0:
        n_rows += 1

    fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols,
                           figsize=(24, (n_rows + 1) * 4))

    # axs = ax.ravel()

    for i, radius in enumerate(range(radius_min, radius_max, radius_step)):
        sns.countplot(
            find_points_within_search_radius(
                calculate_and_sort_square_distance_matrix(
                    coordinates[subset]),
                radius,
                False),
            ax=ax)
        ax.set_title(f"Search radius = {radius} m")
        ax.xaxis.set_tick_params(rotation=0)
    plt.show()


def interactive_plot_search_radius(coordinates, subset=["X", "Y"],
                                   radius_default=500, radius_min=0,
                                   radius_max=3000, radius_step=100,
                                   radius_offset=100, orient='cols',):

    if orient == 'rows':
        n_rows = 3
        n_cols = 1
        fig_size = (24, 12)
    elif orient == 'cols':
        n_rows = 1
        n_cols = 3
        fig_size = (24, 6)
    else:
        raise ValueError("'orient' should be either 'rows' or 'cols'")

    colorpalette = create_color_dict(coordinates)

    @interact(search_radius=(radius_min, radius_max, radius_step))
    def interactive_plot(search_radius=radius_default):
        fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols,
                               figsize=fig_size, sharey=True, sharex=False)

        axs = ax.ravel()

        for i, radius in enumerate([search_radius - radius_offset,
                                    search_radius,
                                    search_radius + radius_offset]):
            axer = sns.countplot(
                        find_points_within_search_radius(
                            calculate_and_sort_square_distance_matrix(
                                coordinates[subset]
                            ),
                            radius,
                            False
                        ),
                        ax=axs[i],
                        palette=colorpalette
                    )

            for p in axer.patches:
                annotate_plot(p, axer)

            axs[i].set_title(f"Search radius = {radius} m", size=12)

            if n_cols == 1:
                axs[-1].set_xlabel("n closest points", size=12)
            else:
                axs[i].set_xlabel("n closest points", size=12)

            if n_rows == 1:
                axs[0].set_ylabel("number\nof\nsamples",
                                  size=12, rotation=0, labelpad=30)
                if i != 0:
                    axs[i].set_ylabel("",
                                      size=12, rotation=0, labelpad=30)
            else:
                axs[i].set_ylabel("number\nof\nsamples",
                                  size=12, rotation=0, labelpad=30)

            axs[i].xaxis.set_tick_params(rotation=45)

            # Change display of spines
            axs[i].spines['right'].set_visible(False)
            axs[i].spines['top'].set_visible(False)

            # Set the color of the visible spines
#             axs[i].spines['bottom'].set_color('grey')


def plot_map_with_control_points(coordinates, pluton, subset=["X", "Y"],
                                 save_fig=True, label_size=5, show_labels=True,
                                 skip_xaxis_label=0, skip_yaxis_label=0,
                                 skip_xaxis_start=0, skip_yaxis_start=0):

    fig, ax = plt.subplots(nrows=1, ncols=1)

    plt.scatter(coordinates[subset[0]],
                coordinates[subset[1]],
                linestyle='None',
                s=5)
    if show_labels:
        for i, row in coordinates.iterrows():
            plt.text(row[subset[0]],
                     row[subset[1]],
                     i,
                     fontsize=label_size)

    if skip_xaxis_label != 0:
        every_nth = skip_xaxis_label
        for n, label in enumerate(ax.xaxis.get_ticklabels(),
                                  start=skip_xaxis_start):
            if n % every_nth != 0:
                label.set_visible(False)

    if skip_yaxis_label != 0:
        every_nth = skip_yaxis_label
        for n, label in enumerate(ax.yaxis.get_ticklabels(),
                                  start=skip_yaxis_start):
            if n % every_nth != 0:
                label.set_visible(False)

    # Set the color of the visible spines
    for spine in ['left', 'right', 'top', 'bottom']:
        ax.spines[spine].set_color('grey')

    # Set general tick parameters
    ax.tick_params(axis='both',
                   direction='in',
                   colors='grey',
                   labelsize=9)

    ax.set_aspect('equal', adjustable='box')

    fig.patch.set_facecolor('white')

    plt.tight_layout()

    if save_fig:
        plt.savefig(f"../_FIGURES/maps_control_points/{pluton}_control_points.pdf")
    plt.show()


def plot_qapf_map(coordinates, pluton, subset=["X", "Y"], qapf="QAPF",
                  save_fig=True, label_size=5, marker_size=12,
                  legend_outside_plot=False, skip_xaxis_label=0,
                  skip_yaxis_label=0, skip_xaxis_start=0, skip_yaxis_start=0,
                  multiplier=0.75, bbox_x_anchor=1.25, plot_labels=True):

    fig, ax = plt.subplots()

    color_palette_custom = {}

    for key, value in zip(QAPF_upper_regions_numbers.keys(),
                          QAPF_upper_regions_numbers_vs_cmap.values()):
        color_palette_custom[key] = create_custom_colormap()[0](value - 0.05)

    # Get correctly ordered list of QAPF classification levels present
    # in df and store it for later use as hue_order in sns.scatterplot
    qapf_unique_list = list(coordinates[qapf].unique())
    qapf_numbers_unique_list = \
        [QAPF_upper_regions_numbers[qapf] for qapf in qapf_unique_list]
    qapf_numbers_unique_list_sorted = np.sort(qapf_numbers_unique_list)
    hue_order_list = \
        [QAPF_upper_regions_numbers_inverse[number]
         for number in qapf_numbers_unique_list_sorted]
    # print(hue_order_list)

    ax = sns.scatterplot(subset[0],
                         subset[1],
                         data=coordinates,
                         s=marker_size,
                         edgecolor=None,
                         hue=qapf,
                         legend='brief',
                         palette=color_palette_custom,
                         hue_order=hue_order_list,)
    if plot_labels:
        for i, row in coordinates.iterrows():
            ax.text(row[subset[0]],
                    row[subset[1]],
                    i,
                    fontsize=label_size)

    plt.gca().set_aspect('equal', adjustable='box')

    if skip_xaxis_label != 0:
        every_nth = skip_xaxis_label
        for n, label in enumerate(ax.xaxis.get_ticklabels(),
                                  start=skip_xaxis_start):
            if n % every_nth != 0:
                label.set_visible(False)

    if skip_yaxis_label != 0:
        every_nth = skip_yaxis_label
        for n, label in enumerate(ax.yaxis.get_ticklabels(),
                                  start=skip_yaxis_start):
            if n % every_nth != 0:
                label.set_visible(False)

    # Set the color of the visible spines
    for spine in ['left', 'right', 'top', 'bottom']:
        ax.spines[spine].set_color('grey')

    # Set general tick parameters
    ax.tick_params(axis='both',
                   direction='in',
                   colors='grey',
                   labelsize=9)

    # Disable axis labels
    ax.set_xlabel("")
    ax.set_ylabel("")

    ax.set_aspect('equal', adjustable='box')

    fig.patch.set_facecolor('white')

    handles, labels = ax.get_legend_handles_labels()

    if legend_outside_plot:
        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * multiplier, box.height])

        # # Put a legend below current axis
        # Slice the handles and labels to remove automatic Seaborn
        # legend title
        ax.legend(handles=handles[1:], labels=labels[1:], loc='upper center',
                  bbox_to_anchor=(bbox_x_anchor, 1),
                  fancybox=True, shadow=True, ncol=1, title=None,
                  fontsize='x-small')
    else:
        # Slice the handles and labels to remove automatic Seaborn
        # legend title
        ax.legend(handles=handles[1:], labels=labels[1:], ncol=1, title="",
                  prop={'size': 8})
        plt.tight_layout()

    if save_fig:
        if plot_labels:
            plt.savefig(f"../_FIGURES/qapf_control_map/{pluton}_qapf_control_with_labels.pdf")
        else:
            plt.savefig(f"../_FIGURES/qapf_control_map/{pluton}_qapf_control.pdf")
    plt.show()


def plot_contour_map(PC_interpolated,
                     coordinates_grid,
                     coordinates_utm,
                     pluton,
                     variable='estimates',
                     levels=None,
                     title="",
                     show_plot=False,
                     single_mineral=False,
                     mineral=None,
                     label_pos=(0.05, 0.95),
                     skip_xaxis_label=0,
                     skip_yaxis_label=0,
                     skip_xaxis_start=0,
                     skip_yaxis_start=0,
                     **kwargs,
                     ):
    fig, ax = plt.subplots(**kwargs)

    if single_mineral:
        to_plot = PC_interpolated
    elif variable == "estimates":
        to_plot = PC_interpolated[0]
    elif variable == "variance":
        to_plot = PC_interpolated[1]
    elif variable == "nvalues":
        to_plot = PC_interpolated[2].reshape(PC_interpolated[0].shape)
    else:
        raise ValueError
    print(mineral)
    contour = ax.contourf(coordinates_grid[0],
                          coordinates_grid[1],
                          to_plot,
                          levels=None,
                          # cmap='viridis')
                          cmap=cmap_dict[mineral])

    # print(coordinates_grid[0].shape,
    #       coordinates_grid[1].shape,
    #       to_plot.shape)

    ax.plot(coordinates_utm["X"].values,
            coordinates_utm["Y"].values,
            'ko', ms=1)

    fig.text(label_pos[0], label_pos[1], mineral,
             transform=ax.transAxes, color='grey')

    ax.set_aspect('equal', adjustable='box')

    cbar = colorbar(contour, ax=ax)
    cbar.set_ticklabels([f"{level.get_text()}%" for level in cbar.ax.get_yticklabels()])
    # cbar.set_ticks(levels)
    # cbar.set_ticklabels([f"{level}%" for level in levels])
    # plt.setp(axs[i].get_xticklabels(), ha="center", rotation=90)

    if skip_xaxis_label != 0:
        every_nth = skip_xaxis_label
        for n, label in enumerate(ax.xaxis.get_ticklabels(),
                                  start=skip_xaxis_start):
            if n % every_nth != 0:
                label.set_visible(False)

    if skip_yaxis_label != 0:
        every_nth = skip_yaxis_label
        for n, label in enumerate(ax.yaxis.get_ticklabels(),
                                  start=skip_yaxis_start):
            if n % every_nth != 0:
                label.set_visible(False)

    # every_nth = 2
    # for n, label in enumerate(ax.xaxis.get_ticklabels()):
    #     if n % every_nth != 0:
    #         label.set_visible(False)

    # Set the color of the visible spines
    for spine in ['left', 'right', 'top', 'bottom']:
        ax.spines[spine].set_color('grey')

    # Set general tick parameters
    ax.tick_params(axis='both',
                   direction='in',
                   colors='grey',
                   labelsize=9)

    # COLORBAR
    # set colorbar label plus label color
    # cbar.set_label("None", color='grey')

    # set colorbar tick color
    cbar.ax.yaxis.set_tick_params(color='grey')

    # set colorbar edgecolor
    cbar.outline.set_edgecolor('grey')

    # set colorbar ticklabels
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='grey')

    # Change display of spines
    # axs[i].spines['right'].set_visible(False)
    # axs[i].spines['top'].set_visible(False)
    # fig.suptitle(f"{pluton} utm {title}")
    fig.patch.set_facecolor('white')
    plt.tight_layout()

    plt.savefig(f"../_FIGURES/contour_maps_single/{pluton}/{pluton}_utm_contourplot_{title.replace(' ', '_')}.pdf")
    if show_plot:
        plt.show()
    else:
        plt.close()


def colorbar(mappable, ax):
    """https://stackoverflow.com/questions/29516157/set-equal-aspect-in-plot-with-colorbar"""
    # ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax)


def plot_contour_map_all(PC_interpolated,
                         coordinates_grid,
                         coordinates_utm,
                         pluton,
                         levels=None,
                         title="",
                         show_plot=False,
                         label_pos=(0.05, 0.95),
                         skip_xaxis_label=0,
                         skip_yaxis_label=0,
                         skip_xaxis_start=0,
                         skip_yaxis_start=0,
                         **kwargs,
                         ):

    # levels = [level for level in range(0, 101, 10)]

    fig, ax = plt.subplots(**kwargs)

    axs = ax.ravel()

    for i, (mineral, interpolation) in enumerate(PC_interpolated.items()):

        contour = axs[i].contourf(coordinates_grid[0],
                                  coordinates_grid[1],
                                  interpolation,
                                  levels=None,
                                  # cmap='viridis')
                                  cmap=cmap_dict[mineral])

        # print(coordinates_grid[0].shape,
        #       coordinates_grid[1].shape,
        #       to_plot.shape)

        axs[i].plot(coordinates_utm["X"].values,
                    coordinates_utm["Y"].values,
                    'ko', ms=1)

        fig.text(label_pos[0], label_pos[1], mineral,
                 transform=axs[i].transAxes, color='grey')

        axs[i].set_aspect('equal', adjustable='box')

        cbar = colorbar(contour, ax=axs[i])
        cbar.set_ticklabels([f"{level.get_text()}%"
                            for level in cbar.ax.get_yticklabels()])
        # cbar.set_ticks(levels)
        # cbar.set_ticklabels([f"{level}%" for level in levels])
        # plt.setp(axs[i].get_xticklabels(), ha="center", rotation=90)

        if skip_xaxis_label != 0:
            every_nth = skip_xaxis_label
            for n, label in enumerate(axs[i].xaxis.get_ticklabels(),
                                      start=skip_xaxis_start):
                if n % every_nth != 0:
                    label.set_visible(False)

        if skip_yaxis_label != 0:
            every_nth = skip_yaxis_label
            for n, label in enumerate(axs[i].yaxis.get_ticklabels(),
                                      start=skip_yaxis_start):
                if n % every_nth != 0:
                    label.set_visible(False)

        # Set the color of the visible spines
        for spine in ['left', 'right', 'top', 'bottom']:
            axs[i].spines[spine].set_color('grey')

        # Set general tick parameters
        axs[i].tick_params(axis='both',
                           direction='in',
                           colors='grey',
                           labelsize=9)

        # COLORBAR
        # set colorbar label plus label color
        # cbar.set_label("None", color='grey')

        # set colorbar tick color
        cbar.ax.yaxis.set_tick_params(color='grey')

        # set colorbar edgecolor
        cbar.outline.set_edgecolor('grey')

        # set colorbar ticklabels
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='grey')

        # Change display of spines
        # axs[i].spines['right'].set_visible(False)
        # axs[i].spines['top'].set_visible(False)

    # fig.suptitle(f"{pluton} utm {title}")
    fig.patch.set_facecolor('white')
    plt.tight_layout()

    plt.savefig(f"../_FIGURES/contour_maps_all/{pluton}_utm_contourplot_all_{title.replace(' ', '_')}.pdf")
    if show_plot:
        plt.show()
    else:
        plt.close()


def create_custom_colormap(base_cmap="tab20"):
    color_list = []

    for i in np.arange(0, 1, 0.05):
        # print(plt.get_cmap(base_cmap)(i))
        color_list.append(plt.get_cmap(base_cmap)(i))

    color_list.insert(0, (1.0, 1.0, 1.0, 1.0))

    cmap_custom = ListedColormap(color_list)
    boundaries = [i for i in range(21)]
    norm_custom = BoundaryNorm(boundaries, cmap_custom.N, clip=True)

    return cmap_custom, norm_custom


def plot_contour_map_interpolated_QAPF(interpolation_array,
                                       grid,
                                       coordinates_utm_qapf,
                                       pluton,
                                       values_to_plot,
                                       number_of_classes=21,
                                       skip_xaxis_label=0,
                                       skip_yaxis_label=0,
                                       skip_xaxis_start=0,
                                       skip_yaxis_start=0,
                                       legend_outside_plot=False,
                                       multiplier=0.75,
                                       bbox_x_anchor=1.3,
                                       plot_control_points=True,
                                       show_qapf_control_points=True,
                                       marker_size=1,
                                       no_legend=False,
                                       **kwargs):

    cmap_custom, norm_custom = create_custom_colormap()
    levels_custom = [i for i in range(number_of_classes + 1)]

    # Add one to incorporate "0: np.nan qapf class"
    number_of_colors = len(values_to_plot) + 1

    values_to_plot_extended = [0]
    values_to_plot_extended.extend(values_to_plot)

    converted_numbers = {}
    converted_numbers_inverse = {}
    cmap_custom_converted = {}

    levels_custom_converted = [i for i in range(number_of_colors + 1)]

    for index, item in enumerate(values_to_plot_extended):
        converted_numbers[item] = index
        converted_numbers_inverse[index] = item
        if item == 0:
            correction = 1
        else:
            correction = 0
        cmap_custom_converted[index] = \
            cmap_custom(QAPF_upper_regions_numbers_vs_cmap[item -
                        1 + correction])
    # cmap_custom_converted[index + 1] = \
    #     cmap_custom(QAPF_upper_regions_numbers_vs_cmap[item])

    cmap_custom_converted = \
        ListedColormap(list(cmap_custom_converted.values()))

    norm_custom_converted = BoundaryNorm(levels_custom_converted,
                                         number_of_colors + 1,
                                         clip=True)

    interpolation_array_converted = \
        np.vectorize(converted_numbers.get)(interpolation_array)

    # Add 1 to every value in array to make sure right color is plotted
    # later on
    interpolation_array_converted += 1

    fig, ax = plt.subplots(**kwargs)

    contour = ax.contourf(grid[0], grid[1], interpolation_array_converted,
                          cmap=cmap_custom_converted,
                          norm=norm_custom_converted,
                          levels=levels_custom_converted)

    if plot_control_points:
        if show_qapf_control_points:
            for index, row in coordinates_utm_qapf.iterrows():
                ax.plot(row["X"],
                        row["Y"],
                        color=cmap_custom(QAPF_upper_regions_numbers_vs_cmap[QAPF_upper_regions_numbers[row["QAPF"]] - 1]),
                        marker='o',
                        markeredgecolor='k',
                        markeredgewidth=0.5,
                        ms=2,
                        linestyle='None')
        else:
            ax.plot(coordinates_utm_qapf["X"],
                    coordinates_utm_qapf["Y"],
                    color='k',
                    marker='o',
                    ms=marker_size,
                    linestyle='None')

    # plt.colorbar(contour)

    if not no_legend:

        # proxy = [plt.Rectangle((0, 0), 1, 1,
        #          fc=cmap_custom(QAPF_upper_regions_numbers_vs_cmap[value]))
        #          for value in values_to_plot]

        # print(contour.collections)

        # Get rectangles with correct color for legend
        proxy = [plt.Rectangle((0, 0), 1, 1, fc=pc.get_facecolor()[0])
                 for pc in contour.collections]

        # for pc in contour.collections:
        #     print(pc.get_facecolor())

        # print(contour.levels)

        values_to_plot_converted = list(converted_numbers.values())

        values = [level for level in contour.levels if level in values_to_plot_converted]
        # print(values)
        values = [value for value in map(int, values)]

        # print(proxy)
        proxy_items = [proxy[value] for value in values_to_plot_converted[1:]]
        # print(proxy_items)
        labels = [QAPF_upper_regions_numbers_inverse[value]
                  for value in values_to_plot]
        # print(values_to_plot)
        # labels = [QAPF_upper_regions_numbers_inverse[value]
        #           for value in values_to_plot]

        # plt.legend(proxy_items, labels)

        if legend_outside_plot:
            # Shrink current axis by 20%
            box = ax.get_position()
            ax.set_position([box.x0,
                             box.y0,
                             box.width * multiplier,
                             box.height,
                             ])

            # # Put a legend below current axis
            ax.legend(proxy_items, labels, loc='upper center',
                      bbox_to_anchor=(bbox_x_anchor, 1),
                      fancybox=True, shadow=True, ncol=1, title=None,
                      fontsize='x-small')
        else:
            ax.legend(proxy_items, labels, ncol=1, title="", prop={'size': 8})
            plt.tight_layout()

    if skip_xaxis_label != 0:
        every_nth = skip_xaxis_label
        for n, label in enumerate(ax.xaxis.get_ticklabels(),
                                  start=skip_xaxis_start):
            if n % every_nth != 0:
                label.set_visible(False)

    if skip_yaxis_label != 0:
        every_nth = skip_yaxis_label
        for n, label in enumerate(ax.yaxis.get_ticklabels(),
                                  start=skip_yaxis_start):
            if n % every_nth != 0:
                label.set_visible(False)

    # Set the color of the visible spines
    for spine in ['left', 'right', 'top', 'bottom']:
        ax.spines[spine].set_color('grey')

    # Set general tick parameters
    ax.tick_params(axis='both',
                   direction='in',
                   colors='grey',
                   labelsize=9)

    ax.set_aspect('equal', adjustable='box')

    fig.patch.set_facecolor('white')

    if show_qapf_control_points:
        plt.savefig(f"../_FIGURES/qapf_contour/{pluton}_QAPF_interpolated_control.pdf")
    else:
        plt.savefig(f"../_FIGURES/qapf_contour/{pluton}_QAPF_interpolated.pdf")

    plt.show()


# def plot_contour_map_interpolated_QAPF_old(interpolation_array,
#                                            grid,
#                                            pluton,
#                                            values_to_plot,
#                                            indices_for_legend,
#                                            number_of_classes=21,
#                                            skip_xaxis_label=0,
#                                            skip_yaxis_label=0):
#     fig, ax = plt.subplots()

#     cmap_custom, norm_custom = create_custom_colormap()

#     levels_custom = values_to_plot.copy()
#     levels_custom = [x * 2 for x in levels_custom]
#     levels_custom_new = []
#     print(levels_custom)
#     for value in levels_custom:
#         if value not in levels_custom_new:
#             levels_custom_new.append(value)
#         levels_custom_new.append(value + 1)
#         if value != 0:
#             levels_custom_new.append(value + 2)
#     # levels_custom_new.insert(0, 9)
#     levels_custom_new.insert(0, 1)
#     levels_custom_new.insert(0, 0)
#     # levels_custom.append(max(values_to_plot) + 1)
#     print(levels_custom_new)

#     # levels_custom = [i for i in range(number_of_classes + 1)]

#     contour = ax.contourf(grid[0], grid[1], interpolation_array,
#                           cmap=cmap_custom, norm=norm_custom,
#                           levels=levels_custom_new)
#     plt.colorbar(contour)

#     indices_for_colors = [levels_custom_new[index] for index in indices_for_legend]

#     proxy = [plt.Rectangle((0, 0), 1, 1,
#              fc=cmap_custom(QAPF_upper_regions_numbers_vs_cmap[value]))
#              for value in indices_for_colors]

#     # proxy = [plt.Rectangle((0, 0), 1, 1, fc=pc.get_facecolor()[0])
#     #          for pc in contour.collections]

#     # values = [level for level in contour.levels if level in values_to_plot]
#     # values = [value - 1 for value in map(int, values)]

#     print(len(proxy))
#     # proxy_items = [proxy[value] for value in indices_for_legend]
#     # labels = [QAPF_upper_regions_numbers_inverse[value + 1]
#     #           for value in values]
#     print(values_to_plot)
#     labels = [QAPF_upper_regions_numbers_inverse[value]
#               for value in values_to_plot]

#     # plt.legend(proxy_items, labels)
#     plt.legend(proxy, labels)

#     if skip_xaxis_label != 0:
#         every_nth = skip_xaxis_label
#         for n, label in enumerate(ax.xaxis.get_ticklabels()):
#             if n % every_nth != 0:
#                 label.set_visible(False)

#     if skip_yaxis_label != 0:
#         every_nth = skip_yaxis_label
#         for n, label in enumerate(ax.yaxis.get_ticklabels(), start=-1):
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
#     plt.tight_layout()

#     plt.savefig(f"../_FIGURES/qapf_contour/{pluton}_QAPF_interpolated.pdf")
#     plt.show()


def custom_colorbar():
    fig, ax = plt.subplots(figsize=(6, 1))
    fig.subplots_adjust(bottom=0.5)

    # cmap = mpl.cm.Reds
    # cmap = mpl.cm.Blues
    cmap = mpl.cm.binary
    # cmap = mpl.cm.Greens
    norm = mpl.colors.Normalize(vmin=0, vmax=100)

    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                    norm=norm,
                                    orientation='horizontal')
    cb1.set_label('Some Units')
    fig.show()


def biplot_control_and_grid(pca_df, pluton, PCx="PC01", PCy="PC02",
                            offsets=[0., 0., 0., 0.],
                            extra_ticks=[[], [], [], []],
                            grid_point_size=1,
                            xspacing=1.0,
                            xformat="2.0f",
                            yspacing=0.5,
                            yformat="2.1f",
                            adjust='datalim',
                            no_legend=False,
                            skip_xaxis_label=0,
                            skip_xaxis_start=0,
                            skip_yaxis_label=0,
                            skip_yaxis_start=0,
                            override_data_boundaries=False,):

    control_points = pca_df.query("Type == 'Control points'")
    grid_points = pca_df.query("Type == 'Grid points'")

    # ax = sns.scatterplot(x=PCx, y=PCy,
    #                      data=grid_points,
    #                      label="Grid predictions",
    #                      color='orangered',
    #                      size=0.1,
    #                      edgecolor=None,
    #                      marker=',')

    ax = plt.scatter(grid_points[PCx],
                     grid_points[PCy],
                     label="Grid points",
                     color='orangered',
                     s=grid_point_size,
                     lw=0,
                     marker='.',
                     linestyle='None')

    ax = sns.scatterplot(x=PCx, y=PCy,
                         data=control_points,
                         s=10,
                         label="Control points",
                         color='k',
                         edgecolor=None,
                         legend=False,
                         )
    if not no_legend:
        handles, labels = ax.get_legend_handles_labels()
        handles = [handles[-1], handles[0]]
        labels = [labels[-1], labels[0]]
        lgnd = ax.legend(handles, labels, loc='best',)

        # Set fixed marker size for both entries in legend
        for i, handle in enumerate(lgnd.legendHandles):
            if i == 0:
                handle.set_sizes([15.0])
            if i == 1:
                handle.set_sizes([90.0])

    axes_labels(PCx.replace("0", ""), PCy.replace("0", ""))

    sns.despine()

    axes_format(ax, pca_df[PCx], pca_df[PCy], offsets,
                adjust=adjust,
                xspacing=xspacing, xformat=xformat,
                yspacing=yspacing, yformat=yformat,
                skip_xaxis_label=skip_xaxis_label,
                skip_xaxis_start=skip_xaxis_start,
                skip_yaxis_label=skip_yaxis_label,
                skip_yaxis_start=skip_yaxis_start,
                extra_ticks=extra_ticks,
                override_data_boundaries=override_data_boundaries)

    # Set facecolor of figure
    plt.gcf().set_facecolor('white')
    plt.tight_layout()

    if adjust == 'box':
        plt.savefig(f"../_FIGURES/biplots/{pluton[:2]}/{pluton}_biplot_control_and_grid_cropped.png",
                    dpi=600,
                    bbox_inches='tight',
                    pad_inches=0)
        plt.savefig(f"../_FIGURES/biplots/{pluton[:2]}/{pluton}_biplot_control_and_grid_cropped.pdf")
    else:
        plt.savefig(f"../_FIGURES/biplots/{pluton[:2]}/{pluton}_biplot_control_and_grid.png",
                    dpi=600,
                    bbox_inches='tight',
                    pad_inches=0)
        plt.savefig(f"../_FIGURES/biplots/{pluton[:2]}/{pluton}_biplot_control_and_grid.pdf")
    plt.show()


def biplot_control_and_grid_technicolor(pca_df, pluton, qapf_regions=None,
                                        PCx="PC01", PCy="PC02",
                                        offsets=[0., 0., 0., 0.],
                                        extra_ticks=[[], [], [], []],
                                        grid_point_size=1,
                                        xspacing=1.0,
                                        xformat="2.0f",
                                        yspacing=0.5,
                                        yformat="2.1f",
                                        adjust='datalim',
                                        no_legend=False,
                                        skip_xaxis_label=0,
                                        skip_xaxis_start=0,
                                        skip_yaxis_label=0,
                                        skip_yaxis_start=0,
                                        override_data_boundaries=False,
                                        test=None,
                                        axes_offset=0.1,):

    control_points = pca_df.query("Type == 'Control points'")
    grid_points = pca_df.query("Type == 'Grid points'")

    color_palette_custom = {}

    # for key, value in zip(QAPF_upper_regions_numbers.keys(),
    #                       QAPF_upper_regions_numbers_vs_cmap.values()):
    #     color_palette_custom[key] = create_custom_colormap()[0](value - 0.05)

    for key, value in QAPF_upper_regions_numbers_vs_cmap.items():
        # print(key, value)
        color_palette_custom[key] = create_custom_colormap()[0](value - 0.05)

    # Get correctly ordered list of QAPF classification levels present
    # in df and store it for later use as hue_order in sns.scatterplot
    qapf_numbers_unique_list = list(grid_points["QAPF_numbers"].unique())
    qapf_numbers_unique_list_sorted = np.sort(qapf_numbers_unique_list)

    # ax = sns.scatterplot(x=PCx, y=PCy,
    #                      data=grid_points,
    #                      label="Grid predictions",
    #                      color='orangered',
    #                      size=0.1,
    #                      edgecolor=None,
    #                      marker=',')

    filtered_color_map = list(color_palette_custom.values())
    filtered_color_map = \
        [filtered_color_map[index] for index in
         qapf_numbers_unique_list]

    ax = plt.scatter(grid_points[PCx],
                     grid_points[PCy],
                     label="Grid points",
                     c=grid_points["QAPF_colors"],
                     # cmap=ListedColormap(filtered_color_map),
                     s=grid_point_size,
                     lw=0,
                     marker='.',
                     linestyle='None')

    ax = sns.scatterplot(x=PCx, y=PCy,
                         data=control_points,
                         s=10,
                         label="Control points",
                         hue="QAPF_numbers",
                         hue_order=qapf_numbers_unique_list_sorted,
                         palette=color_palette_custom,
                         edgecolor='k',
                         linewidth=0.3,
                         legend=False,
                         )

    if qapf_regions is not None:
        for region_nr in qapf_numbers_unique_list_sorted:
            # print(region_nr)
            qapf_region_to_plot = qapf_regions.query("QAPF_numbers == @region_nr")
            print(qapf_region_to_plot.loc[:, "QAPF_numbers"])

            ax.scatter(qapf_region_to_plot[PCx],
                       qapf_region_to_plot[PCy],
                       # label="QAPF regions",
                       color='k',
                       s=5,
                       marker='o',
                       linestyle="-",
                       linewidth=2)

    if test is not None:
        ax.scatter(test[PCx],
                   test[PCy],
                   color='r',
                   s=5,
                   marker='o')

    if not no_legend:
        handles, labels = ax.get_legend_handles_labels()
        handles = [handles[-1], handles[0]]
        labels = [labels[-1], labels[0]]
        lgnd = ax.legend(handles, labels, loc='best',)

        # Set fixed marker size for both entries in legend
        for i, handle in enumerate(lgnd.legendHandles):
            if i == 0:
                handle.set_sizes([15.0])
            if i == 1:
                handle.set_sizes([90.0])

    axes_labels(PCx.replace("0", ""), PCy.replace("0", ""))

    sns.despine()

    axes_format(ax, pca_df[PCx], pca_df[PCy], offsets,
                adjust=adjust,
                xspacing=xspacing, xformat=xformat,
                yspacing=yspacing, yformat=yformat,
                skip_xaxis_label=skip_xaxis_label,
                skip_xaxis_start=skip_xaxis_start,
                skip_yaxis_label=skip_yaxis_label,
                skip_yaxis_start=skip_yaxis_start,
                extra_ticks=extra_ticks,
                override_data_boundaries=override_data_boundaries,
                axes_offset=axes_offset,)

    # Set facecolor of figure
    plt.gcf().set_facecolor('white')
    plt.tight_layout()

    if adjust == 'box':
        plt.savefig(f"../_FIGURES/biplots/{pluton[:2]}/{pluton}_biplot_control_and_grid_cropped_technicolor.png",
                    dpi=900,
                    bbox_inches='tight',
                    pad_inches=0)
        plt.savefig(f"../_FIGURES/biplots/{pluton[:2]}/{pluton}_biplot_control_and_grid_cropped_technicolor.pdf")
    else:
        plt.savefig(f"../_FIGURES/biplots/{pluton[:2]}/{pluton}_biplot_control_and_grid_technicolor.png",
                    dpi=900,
                    bbox_inches='tight',
                    pad_inches=0)
        plt.savefig(f"../_FIGURES/biplots/{pluton[:2]}/{pluton}_biplot_control_and_grid_technicolor.pdf")
    plt.show()


color_mapping = create_custom_colormap()[0]


def get_colors(x):
    color = color_mapping(x - 0.05)
    return color


def biplot_control_and_solutions(pca_df, pluton, PCx="PC01", PCy="PC02",
                                 offsets=[0., 0., 0., 0.],
                                 extra_ticks=[[], [], [], []],
                                 xspacing=1.0,
                                 xformat="2.0f",
                                 yspacing=0.5,
                                 yformat="2.1f",
                                 adjust='datalim',
                                 no_legend=False,
                                 skip_xaxis_label=0,
                                 skip_xaxis_start=0,
                                 skip_yaxis_label=0,
                                 skip_yaxis_start=0,
                                 override_data_boundaries=False,
                                 axes_offset=0.1,):

    control_points = pca_df.query("Type == 'Control points'")
    grid_points = pca_df.query("Type == 'CV predictions'")

    if not no_legend:
        legend = 'brief'
    else:
        legend = False

    ax = sns.scatterplot(x=PCx, y=PCy,
                         data=grid_points,
                         label="CV predictions",
                         color='tab:green',
                         s=25,
                         legend=legend)

    ax = sns.scatterplot(x=PCx, y=PCy,
                         data=control_points,
                         label="Control points",
                         color='k',
                         s=25,
                         legend=legend)
    if not no_legend:
        handles, labels = ax.get_legend_handles_labels()
        handles = [handles[-1], handles[0]]
        labels = [labels[-1], labels[0]]
        ax.legend(handles, labels, loc='best')

    axes_labels(PCx.replace("0", ""), PCy.replace("0", ""))

    sns.despine()

    axes_format(ax, pca_df[PCx], pca_df[PCy], offsets,
                adjust=adjust,
                xspacing=xspacing, xformat=xformat,
                yspacing=yspacing, yformat=yformat,
                skip_xaxis_label=skip_xaxis_label,
                skip_xaxis_start=skip_xaxis_start,
                skip_yaxis_label=skip_yaxis_label,
                skip_yaxis_start=skip_yaxis_start,
                extra_ticks=extra_ticks,
                override_data_boundaries=override_data_boundaries,
                axes_offset=axes_offset,)

    # Set facecolor of figure
    plt.gcf().set_facecolor('white')
    plt.tight_layout()

    if adjust == 'box':
        plt.savefig(f"../_FIGURES/biplots/{pluton[:2]}/{pluton}_biplot_control_and_solutions_cropped.pdf")
    else:
        plt.savefig(f"../_FIGURES/biplots/{pluton[:2]}/{pluton}_biplot_control_and_solutions.pdf")
    plt.show()


def biplot_residuals(pca_df, pluton, PCx="PC01", PCy="PC02",
                     offsets=[0., 0., 0., 0.],
                     extra_ticks=[[], [], [], []],
                     xspacing=1.0,
                     xformat="2.0f",
                     yspacing=0.5,
                     yformat="2.1f",
                     adjust='datalim',
                     no_legend=False,
                     skip_xaxis_label=0,
                     skip_xaxis_start=0,
                     skip_yaxis_label=0,
                     skip_yaxis_start=0,
                     override_data_boundaries=False,
                     axes_offset=0.1,
                     **kwargs,):

    if not no_legend:
        legend = 'brief'
    else:
        legend = False

    fig, ax = plt.subplots(**kwargs)

    ax = sns.scatterplot(x=PCx, y=PCy,
                         data=pca_df,
                         label="Residuals",
                         color='tab:blue',
                         s=25,
                         legend=legend)
    if not no_legend:
        handles, labels = ax.get_legend_handles_labels()
        # handles = [handles[-1], handles[0]]
        # labels = [labels[-1], labels[0]]
        ax.legend(handles, labels, loc='best')

    axes_labels(PCx.replace("0", ""), PCy.replace("0", ""))

    sns.despine()

    axes_format(ax, pca_df[PCx], pca_df[PCy], offsets,
                adjust=adjust,
                xspacing=xspacing, xformat=xformat,
                yspacing=yspacing, yformat=yformat,
                skip_xaxis_label=skip_xaxis_label,
                skip_xaxis_start=skip_xaxis_start,
                skip_yaxis_label=skip_yaxis_label,
                skip_yaxis_start=skip_yaxis_start,
                extra_ticks=extra_ticks,
                override_data_boundaries=override_data_boundaries,
                axes_offset=axes_offset)

    # Set facecolor of figure
    plt.gcf().set_facecolor('white')
    plt.tight_layout()

    if adjust == 'box':
        plt.savefig(f"../_FIGURES/biplots/{pluton[:2]}/{pluton}_biplot_residuals_cropped.pdf")
    else:
        plt.savefig(f"../_FIGURES/biplots/{pluton[:2]}/{pluton}_biplot_residuals.pdf")
    plt.show()


def axes_labels(textx, texty):
    plt.xlabel(textx, color='grey', ha='center', fontsize=10)
    plt.ylabel(texty, color='grey', rotation=0, labelpad=10, va='center',
               fontsize=10)


def create_custom_tick_list(data, spacing=1.0, format="2.0f"):
    data_min = data.min()
    data_max = data.max()

    if spacing < 1.0:
        ticks_start = np.floor(data_min) + spacing
        ticks_end = np.ceil(data_max)
    else:
        ticks_start = np.floor(data_min)
        ticks_end = np.ceil(data_max) + 0.01

    ticks = np.arange(ticks_start,
                      ticks_end,
                      spacing)

    tick_list_formatted = [float(f"{x:{format}}") for x in ticks]

    return tick_list_formatted


def axes_format(ax, datax, datay, offsets,
                adjust='datalim',
                xspacing=1.0, xformat="2.0f",
                yspacing=0.5, yformat="2.1f",
                extra_ticks=[[], [], [], []],
                skip_xaxis_label=0, skip_xaxis_start=0,
                skip_yaxis_label=0, skip_yaxis_start=0,
                override_data_boundaries=False,
                axes_offset=0):

    ax.set_aspect('equal', adjustable=adjust)

    xtick_list_formatted = create_custom_tick_list(datax, xspacing, xformat)
    ytick_list_formatted = create_custom_tick_list(datay, yspacing, yformat)

    if extra_ticks[0]:
        for extra_tick in extra_ticks[0]:
            xtick_list_formatted.insert(0, extra_tick)
    if extra_ticks[1]:
        for extra_tick in extra_ticks[1]:
            xtick_list_formatted.append(extra_tick)
    if extra_ticks[2]:
        for extra_tick in extra_ticks[2]:
            ytick_list_formatted.insert(0, extra_tick)
    if extra_ticks[3]:
        for extra_tick in extra_ticks[3]:
            ytick_list_formatted.append(extra_tick)

    ax.set_xticks(xtick_list_formatted)
    print(xtick_list_formatted)
    ax.set_yticks(ytick_list_formatted)
    print(ytick_list_formatted)

    if skip_xaxis_label != 0:
        every_nth = skip_xaxis_label
        for n, label in enumerate(ax.xaxis.get_ticklabels(),
                                  start=skip_xaxis_start):
            if n % every_nth != 0:
                label.set_visible(False)

    if skip_yaxis_label != 0:
        every_nth = skip_yaxis_label
        for n, label in enumerate(ax.yaxis.get_ticklabels(),
                                  start=skip_yaxis_start):
            if n % every_nth != 0:
                label.set_visible(False)

    xmin_spine_boundary = np.min([datax.min(), xtick_list_formatted[0]])
    xmax_spine_boundary = np.max([datax.max(), xtick_list_formatted[-1]])

    ymin_spine_boundary = np.min([datay.min(), ytick_list_formatted[0]])
    ymax_spine_boundary = np.max([datay.max(), ytick_list_formatted[-1]])

    if override_data_boundaries:
        xmin_spine_boundary = xtick_list_formatted[0]
        xmax_spine_boundary = xtick_list_formatted[-1]

        ymin_spine_boundary = ytick_list_formatted[0]
        ymax_spine_boundary = ytick_list_formatted[-1]

    if adjust == 'box':
        # Set the limits of axes
        ax.set_xlim(xmin_spine_boundary-axes_offset, xmax_spine_boundary+0.1)
        ax.set_ylim(ymin_spine_boundary-axes_offset, ymax_spine_boundary+0.1)

    # Set the bounds of the spines
    ax.spines['bottom'].set_bounds(xmin_spine_boundary-offsets[0],
                                   xmax_spine_boundary+offsets[1])

    ax.spines['left'].set_bounds(ymin_spine_boundary-offsets[2],
                                 ymax_spine_boundary+offsets[3])

    # Set the color of the visible spines
    for spine in ['left', 'right', 'top', 'bottom']:
        ax.spines[spine].set_color('grey')

    # Set general tick parameters
    ax.tick_params(axis='both',
                   direction='in',
                   colors='grey',
                   labelsize=9)


def plot_simulation_results(results, pluton, nugget_theoretical,
                            n_control_points, nugget_replicates=None,
                            xmax=1000, text_offsets=[3, 0.002], fit=None,
                            fit_params=None, fit_mode=None, inverse=False):

    sample_size_start = np.amin(list(results.keys()))
    ymax = np.amax(list(results.values()))
    xmax_adjusted = xmax - sample_size_start

    if inverse:
        results_ = {1/key: value for key, value in results.items()}
        sample_size_start_ = 1 / sample_size_start
    else:
        results_ = results.copy()
        sample_size_start_ = sample_size_start.copy()

    fig, ax = plt.subplots()

    plt.plot(list(results_.keys())[:xmax_adjusted],
             list(results_.values())[:xmax_adjusted],
             linewidth=1,
             color='orangered')

    plt.text(0 + sample_size_start_ + text_offsets[0],
             ymax - 2*text_offsets[1],
             "simulations",
             color="orangered",
             size=8)

    sample_size_on_theoretical_nugget = \
        np.where(
            np.isclose(
                list(results_.values()),
                nugget_theoretical,
                rtol=1e-01))[0][0] +\
        sample_size_start

    x_points = [sample_size_on_theoretical_nugget, n_control_points]
    y_points = [nugget_theoretical, results[n_control_points]]

    text_notes = ["theoretical nugget",
                  "control points"]

    if nugget_replicates is not None:
        sample_size_on_replicates_nugget = \
            np.where(
                np.isclose(
                    list(results_.values()),
                    nugget_replicates,
                    rtol=1e-01))[0][0] +\
            sample_size_start

        x_points.append(sample_size_on_replicates_nugget)
        y_points.append(nugget_replicates)

    if inverse:
        x_points_ = [1/x for x in x_points]
    else:
        x_points_ = x_points.copy()

    text_notes.append("replicates' nugget")

    plt.plot(x_points_,
             y_points,
             marker='+',
             color='k',
             linestyle='None')

    # plt.plot(1/8, results[8], marker='o')

    for x, y, text in zip(x_points_, y_points, text_notes):
        plt.text(x + text_offsets[0],
                 y + text_offsets[1],
                 text,
                 color='k',
                 size=8)

    if fit is not None:
        print(fit[:xmax_adjusted].shape)
        plt.plot(list(results_.keys())[:xmax_adjusted],
                 fit[:xmax_adjusted],
                 linewidth=2,
                 color='b')
        if fit_params is not None:
            if fit_mode == "reciprocal":
                fit_equation = f"$y = \\frac{{{fit_params[0]:4.4f}}}{{x}} + {{{fit_params[1]:6.6f}}}$"
            elif fit_mode == "reciprocal2":
                fit_equation = f"$y = {fit_params[0]:4.4f}*x^{{{fit_params[1]:2.4f}}}$"## + {fit_params[2]:6.6f}$"
            elif fit_mode == "exponential":
                fit_equation = f"$y = {{{fit_params[0]:6.4f}}} * e^{{{{{fit_params[1]:4.4f}}}x}}$"
            elif fit_mode == "linear":
                fit_equation = f"$y = {fit_params[0]:2.4f} * x + {fit_params[1]:2.6f}$"
            elif fit_mode == "linear2":
                fit_equation = f"$y = {fit_params[0]:2.4f} * x^{{{fit_params[1]:2.4f}}} + {fit_params[2]:2.6f}$"
            else:
                fit_equation = None
            plt.text(0 + sample_size_start_ + text_offsets[0],
                     fit.max() - 2*text_offsets[1],
                     fit_equation,
                     color="blue",
                     size=8)

    # # Simulated sample size
    # plt.hlines(nugget_theoretical,
    #            sample_size_on_nugget*0.95,
    #            sample_size_on_nugget*1.05,
    #            linewidth=0.3)

    # plt.vlines(sample_size_on_nugget,
    #            nugget_theoretical*0.9,
    #            nugget_theoretical*1.1,
    #            linewidth=0.3)

    # plt.text(sample_size_on_nugget + text_offset,
    #          nugget_theoretical,
    #          f"{sample_size_on_nugget}")

    # Control points
    # plt.hlines(results[n_control_points],
    #            0,
    #            xmax,
    #            linewidth=0.3)

    # plt.vlines(n_control_points,
    #            0,
    #            ymax,
    #            linewidth=0.3)

    # plt.text(n_control_points,
    #          results[n_control_points],
    #          n_control_points)

    # axes labels
    x_label = plt.xlabel("Sample size", color='grey', ha='center')
    y_label = plt.ylabel("MSD", color='grey', rotation=0, labelpad=10,
                         va='center')
    # y_label.draw()
    # print(y_label)

    x_scaling = 50
    y_scaling = 30

    if inverse:
        xmin = xmax/x_scaling
    else:
        xmin = -xmax/x_scaling
    ymin = -ymax/y_scaling

    if inverse:
        x_scaling = 1 / x_scaling
        xmax_ = 1 / xmax
        xmin_ = 1 / xmin
    else:
        xmax_ = xmax
        xmin_ = xmin

    print(xmin_, xmax_)
    if inverse:
        plt.xlim(xmax_, xmin_)
    else:
        plt.xlim(xmin_, xmax_)
    plt.ylim(ymin, ymax)

    x_axis_range = xmax_ - xmin_
    y_axis_range = ymax - ymin

    # Align axes labels with middle tick
    if inverse:
        pass
    else:
        x_label.set_x((1/x_axis_range) *
                      (x_axis_range/2 + (x_axis_range/2 - xmax_/2)))
    y_label.set_y((1/y_axis_range) *
                  (y_axis_range/2 + (y_axis_range/2 - ymax/2)))

    # Set the color of the visible spines
    for spine in ["left", "right", "top", "bottom"]:
        if spine in ["right", "top"]:
            ax.spines[spine].set_visible(False)
        else:
            ax.spines[spine].set_color('grey')

    # Set the bounds of the spines
    if inverse:
        ax.spines['bottom'].set_bounds(0, xmin_)
    else:
        ax.spines['bottom'].set_bounds(0, xmax_)

    ax.spines['left'].set_bounds(0., ymax)

    # xticks
    if inverse:
        xticks_inverse = [0, xmin_]
        print(xticks_inverse)
        xticks = [0, xmin]
        xticks.extend(x_points)
        xticks = np.sort(xticks)
        xticks_inverse.extend([1/x for x in x_points])
        xticks_inverse = np.sort(xticks_inverse)
        print(xticks_inverse)
        ax.set_xticks(xticks_inverse)
    else:
        xticks = [xmax/2, 0, xmax]
        xticks.extend(x_points)
        xticks = np.sort(xticks)
        ax.set_xticks(xticks)

    print(xticks)
    if inverse:
            xtick_labels = [f"$\\frac{{1}}{{{int(x)}}}$" if x != xmax/2 else None for x in xticks]
            xtick_labels = [xtick_labels[index] if index != 0 else 0 for index in range(len(xtick_labels)-1, -1, -1)]
            xtick_labels = xtick_labels[:-1]
            xtick_labels.insert(0, 0)
    else:
        xtick_labels = [int(x) if x != xmax/2 else None for x in xticks]
    print(xtick_labels)
    ax.set_xticklabels(xtick_labels)

    # yticks
    yticks = [ymax/2, 0., ymax]
    yticks.extend(y_points)
    yticks = np.sort(yticks)
    ytick_exlusions = [ymax/2]
    ytick_exlusions.extend(y_points)
    ytick_labels = [f"{y:2.3f}" if y not in ytick_exlusions
                    else None for y in yticks]
    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels)

    # Set general tick parameters
    ax.tick_params(axis='both',
                   direction='in',
                   colors='grey',
                   labelsize=9)

    # ax.set_aspect('equal', adjustable='box')

    fig.patch.set_facecolor('white')

    plt.tight_layout()

    if inverse:
        inverse_mode = "inverse"
    else:
        inverse_mode = ""

    plt.savefig(f"../_FIGURES/simulations/{pluton}_simulation_results_MSD_{fit_mode}_{inverse_mode}.pdf")
    plt.show()

    return x_points, y_points
