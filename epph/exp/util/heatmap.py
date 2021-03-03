'''
@author: Leila Arras
@maintainer: Leila Arras
@date: 21.06.2017
@version: 1.0+
@copyright: Copyright (c) 2017, Leila Arras, Gregoire Montavon, Klaus-Robert Mueller, Wojciech Samek
@license: see LICENSE file in repository root
'''

import matplotlib.pyplot as plt


def rescale_score_by_abs(score, max_score, min_score):
    """
    Normalize the relevance value (=score), accordingly to the extremal relevance values (max_score and min_score), 
    for visualization with a diverging colormap.
    i.e. rescale positive relevance to the range [0.5, 1.0], and negative relevance to the range [0.0, 0.5],
    using the highest absolute relevance for linear interpolation.
    """

    # CASE 1: positive AND negative scores occur --------------------
    if max_score > 0 and min_score < 0:

        if max_score >= abs(min_score):  # deepest color is positive
            if score >= 0:
                return 0.5 + 0.5 * (score / max_score)
            else:
                return 0.5 - 0.5 * (abs(score) / max_score)

        else:  # deepest color is negative
            if score >= 0:
                return 0.5 + 0.5 * (score / abs(min_score))
            else:
                return 0.5 - 0.5 * (score / min_score)

    # CASE 2: ONLY positive scores occur -----------------------------
    elif max_score > 0 and min_score >= 0:
        if max_score == min_score:
            return 1.0
        else:
            return 0.5 + 0.5 * (score / max_score)

    # CASE 3: ONLY negative scores occur -----------------------------
    elif max_score <= 0 and min_score < 0:
        if max_score == min_score:
            return 0.0
        else:
            return 0.5 - 0.5 * (score / min_score)



def getRGB(c_tuple):
    return "#%02x%02x%02x" % (int(c_tuple[0] * 255), int(c_tuple[1] * 255), int(c_tuple[2] * 255))


def div_event(w, score, colormap, scores_dict_context_attr, idx, max_s, min_s):

    context_attributes = ""
    for i, context_attr in enumerate(scores_dict_context_attr):
        score_context_attr = rescale_score_by_abs(
                scores_dict_context_attr[context_attr][len(scores_dict_context_attr[context_attr]) - idx - 1],
                max_s, min_s)
        context_attributes += get_div(w[i + 1], score_context_attr, colormap, get_context_attr_style())

    output_event_word = get_div(w[0] + context_attributes, score, colormap, get_event_attr_style())
    return output_event_word


def get_div(word, score, colormap, styles):
    if colormap == None:
        rgb = "#ffffff"
    else:
        rgb = getRGB(colormap(score))
    return "{}{}{}{}{}{}{}{}".format("<div ", " style=\"background-color:", rgb, ";", styles, "\">", word, "</div>")


def get_legend(args, preprocessor, scores_dict_context_attr):

    context_attributes = preprocessor.get_context_attributes()
    column_names = [args.activity_key]
    column_names.extend(context_attributes)

    context_attributes_html = ""
    for i in range(len(scores_dict_context_attr)):
        context_attributes_html += get_div(column_names[i + 1], 0, None, get_context_attr_style())
    output_event_word = get_div(column_names[0] + context_attributes_html, 0, None, get_event_attr_style())
    return "<br>" + "Legend: " + output_event_word + "<br>"


def get_context_attr_style():
    return "margin: 5px; padding: 2px; border-style: solid; border-width: 1px; display:inline-block"


def get_event_attr_style():
    return "display:inline-block; border-style: solid; border-width: 1px;"


def html_heatmap(events, R_scores, R_scores_dict_context_attr, cmap_name="bwr"):
    """
    Return attribute-level heatmap in HTML format.
    Events: List of events. An event consists of an activity and one or more context attributes. All attributes are strings.
    R_scores: relevance values for activities
    R_scores_dict_context_attr: relevance values for context attributes
    cmap_name the name of the matplotlib diverging colormap.
    """

    colormap = plt.get_cmap(cmap_name)

    # assert len(words)==len(scores)
    max_context = 0
    min_context = 0
    for context_attr in R_scores_dict_context_attr:
        max_context = max(max_context, max(R_scores_dict_context_attr[context_attr]))
        min_context = min(min_context, min(R_scores_dict_context_attr[context_attr]))

    max_s = max(max(R_scores), max_context)
    min_s = min(min(R_scores), min_context)

    output_text = ""

    for idx, event in enumerate(events):
        score = rescale_score_by_abs(R_scores[len(R_scores) - idx - 1], max_s, min_s)
        output_text += div_event(event, score, colormap, R_scores_dict_context_attr, idx, max_s, min_s) + " "

    return output_text + "\n"


def create_html_heatmap_from_relevance_scores(args, preprocessor, heatmap, R_words_context):
    """
    Creates html heatmap from calculated relevance scores.

    Parameters
    ----------
    heatmap : str
        HTML code from relevance scores.
    R_words_context : dict
        An entry in the dict contains relevance scores of context attributes (key is attribute name, value is array)
    column_names : list of str
        Names of attributes (activity + context) considered in prediction.

    Returns
    -------
    str : html code as string for heatmap

    """
    # create html skeleton
    # in head  "<style>" "</style>" could be placed in order to make div tags able to hover/unfold
    head_and_style = \
        "<!DOCTYPE html> <html lang=\"en\">" \
            "<head> " \
                "<meta charset=\"utf-8\"> " \
                "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0 \">" \
                "<title>XNAP2.0</title> " \
            "</head>" \
            "<body>"

    body_end = \
            "</body>" \
        "</html>"

    # create legend
    legend = get_legend(args, preprocessor, R_words_context)

    return head_and_style + legend + heatmap + body_end


def add_relevance_to_heatmap(heatmap, prefix_words, R_words, R_words_context):
    """

    Parameters
    ----------
    heatmap : str
        HTML code from relevance scores.
    prefix_words : list of lists, where a sublist list contains strings
        Sublists represent single events. Strings in a sublist represent original attribute values of this event.
    R_words : ndarray with shape [1, max case length]
        Relevance scores of events in the subsequence to be explained.
    R_words_context : dict
        An entry in the dict contains relevance scores of context attributes (key is attribute name, value is array)

    Returns
    -------

    """
    heatmap += "<br>" + html_heatmap(prefix_words, R_words, R_words_context) + "<br>"
    return heatmap
