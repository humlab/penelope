# from __future__ import print_function
import ipywidgets as widgets

#from .widgets_config import years_widget

BUTTON_STYLE = dict(description_width='initial', button_color='lightgreen')

def button2(description, style=None, callback=None):
    style = style or dict(description_width='initial', button_color='lightgreen')
    button = widgets.Button(description=description, style=style)
    if callback is not None:
        button.on_click(callback)
    return button

def text_widget(element_id=None, default_value=''):
    value = "<span class='{}'>{}</span>".format(element_id, default_value) if element_id is not None else ''
    return widgets.HTML(value=value, placeholder='', description='')

def next_id_button(that, property_name, count):

    def f(_):
        control = getattr(that, property_name, None)
        if control is not None:
            control.value = (control.value + 1) % count

    return button2(description=">>", callback=f)

def prev_id_button(that, property_name, count):

    def f(_):
        control = getattr(that, property_name, None)
        if control is not None:
            control.value = (control.value - 1) % count

    return button2(description="<<", callback=f)

# class WidgetUtility():

#     def __init__(self, **kwargs):
#         for key, value in kwargs.items():
#             setattr(self, key, value)
#         # self.__dict__.update(kwargs)

# class TopicWidgets(WidgetUtility):

#     def __init__(self, n_topics, years=None, word_count=None, text_id=None):

#         self.n_topics = n_topics
#         self.text_id = text_id
#         self.text = text_widget(text_id)
#         self.year = years_widget(options=years) if years is not None else None
#         self.topic_id = self.topic_id_slider(n_topics)
#         self.word_count = self.word_count_slider(1, 500) if word_count is not None else None
#         self.prev_topic_id = button2(description="<<", callback=self.prev_topic_id_clicked)
#         self.next_topic_id = button2(description=">>", callback=self.next_topic_id_clicked)

#     def next_topic_id_clicked(self, _):
#         self.topic_id.value = (self.topic_id.value + 1) % self.n_topics

#     def prev_topic_id_clicked(self, _):
#         self.topic_id.value = (self.topic_id.value - 1) % self.n_topics


# class TopTopicWidgets(WidgetUtility):

#     def __init__(self, n_topics=0, years=None, aggregates=None, text_id='text_id', layout_algorithms=None):

#         self.n_topics = n_topics
#         self.text_id = text_id
#         self.text = text_widget(text_id) if text_id is not None else None
#         self.year = years_widget(options=years) if years is not None else None

#         self.topics_count = self.topic_count_slider(n_topics) if n_topics > 0 else None

#         self.aggregate = self.select_aggregate_fn_widget(aggregates, default='mean') if aggregates is not None else None
#         self.layout_algorithm = self.layout_algorithm_widget(layout_algorithms, default='Fruchterman-Reingold') \
#             if layout_algorithms is not None else None


#wf = WidgetUtility()
