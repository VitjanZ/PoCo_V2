from PyQt5.QtWidgets import QApplication, QGraphicsItem
from PyQt5.QtGui import QColor, QFont, QPen, QBrush
from PyQt5.QtCore import Qt, QPointF, QRectF


class InProgressAnnotation(QGraphicsItem):
    def __init__(self, position, scene, rect, rectangular=False):
        super().__init__()
        self.x1 = rect[0] if rect[2] > 0 else rect[0] + rect[2]
        self.y1 = rect[1] if rect[3] > 0 else rect[1] + rect[3]
        self.w = abs(rect[2])
        self.h = abs(rect[3])
        self.r = (self.w**2+self.h**2)**0.5
        self.w_rel = rect[2]
        self.h_rel = rect[3]

        self.rect = QRectF(0, 0, rect[2], rect[3])
        self.rectangular = rectangular
        self.setPos(position)
        self.curr_scene = scene
        scene.clearSelection()
        self.color_green = QColor(0, 255, 0)
        self.color_red = QColor(255, 0, 0)
        self.color_blue = QColor(0, 0, 255)
        self.color = self.color_blue

    def boundingRect(self):
        return self.rect

    def paint(self, painter, option, widget):
        painter.setPen(self.color)
        painter.setFont(QFont('Decorative', 10))
        if self.rectangular:
            painter.drawRect(self.rect)
        else:
            painter.drawEllipse(QPointF(self.w_rel, self.h_rel), self.r, self.r)


class AnnotationItem(QGraphicsItem):
    def __init__(self, position, scene, rect, annotation_negative=False):
        super().__init__()

        self.setFlags(QGraphicsItem.ItemIsSelectable)
        self.setAcceptHoverEvents(True)
        self.local_rect = QRectF(0, 0, abs(rect[2]), abs(rect[3]))
        self.x = rect[0] if rect[2] > 0 else rect[0] + rect[2]
        self.y = rect[1] if rect[3] > 0 else rect[1] + rect[3]
        self.w = abs(rect[2])
        self.h = abs(rect[3])
        self.r = (self.w**2+self.h**2)**0.5
        self.w_rel = rect[2]
        self.h_rel = rect[3]
        self.rectangular = annotation_negative
        if not self.rectangular:
            self.local_rect = QRectF(-(2**0.5)*self.w_rel, -(2**0.5)*self.h_rel, 2*self.r, 2*self.r)


        self.setPos(QPointF(self.x, self.y))
        self.curr_scene = scene
        scene.clearSelection()
        self.color_green = QColor("#7FFF00")
        self.color_red = QColor(255, 0, 0)
        self.color_blue = QColor(0, 0, 255)
        self.color = self.color_green
        self.annotation_negative = annotation_negative
        if self.annotation_negative:
            self.color = self.color_red
        self.base_color = self.color

    def handle_hover_event(self, QGraphicsSceneHoverEvent):
        modifiers = QApplication.keyboardModifiers()
        pos = QGraphicsSceneHoverEvent.pos()
        in_circle = True if (pos.x()**2+pos.y()**2)**0.5 <= self.r else False
        if (self.curr_scene.picked_tool == 0 or self.curr_scene.picked_tool == 2) and in_circle:
            if (modifiers == Qt.ControlModifier):
                self.color = self.color_red
            else:
                self.color = self.color_blue
        self.curr_scene.update()

    def hoverEnterEvent(self, QGraphicsSceneHoverEvent):
        self.handle_hover_event(QGraphicsSceneHoverEvent)

    def hoverMoveEvent(self, QGraphicsSceneHoverEvent):
        self.handle_hover_event(QGraphicsSceneHoverEvent)

    def hoverLeaveEvent(self, QGraphicsSceneHoverEvent):
        self.color = self.base_color
        self.curr_scene.update()

    def handle_removal_event(self, removal_from_undo=False):
        # TODO REIMPLEMENT ANNOTATIONS_RECT TO REMOVE FASTER, SET, HASHMAP?
        if not self.annotation_negative:
            self.curr_scene.annotation_manager.annotations_rect[self.curr_scene.image_name].remove(
                (self.x, self.y, self.w, self.h))
            self.curr_scene.annotations.remove(self)
            self.curr_scene.annotation_manager.annotations_changed[self.curr_scene.image_name] = True
        else:
            self.curr_scene.annotation_manager.negative_annotations_rect[self.curr_scene.image_name].remove(
                (self.x, self.y, self.w, self.h))
            self.curr_scene.negative_annotations.remove(self)

        self.curr_scene.removeItem(self)
        self.curr_scene.count_label.setText("Number of annotations: " + str(len(self.curr_scene.annotations)))
        self.curr_scene.annotation_tool.draw_convex_hull()
        if not removal_from_undo:
            self.curr_scene.annotation_tool.annotation_undo_stack.append((0, self))
            self.curr_scene.action_undo_stack.append(self.curr_scene.picked_tool)

    def mousePressEvent(self, QMouseEvent):
        # if left mouse button
        modifiers = QApplication.keyboardModifiers()
        if self.curr_scene.picked_tool == 0 or self.curr_scene.picked_tool == 2:
            if (QMouseEvent.button() == 1 and modifiers == Qt.ControlModifier):
                self.handle_removal_event()

    def boundingRect(self):
        return self.local_rect

    def paint(self, painter, option, widget):
        brush = QBrush(self.color)
        pen = QPen(brush, 3)
        painter.setPen(pen)
        painter.setFont(QFont('Decorative', 10))
        if self.rectangular:
            painter.drawRect(self.local_rect)
        else:
            painter.drawEllipse(QPointF(0, 0), self.r, self.r)
