from PyQt5.QtWidgets import QApplication, QGraphicsItem
from PyQt5.QtGui import QColor, QFont, QPen, QBrush
from PyQt5.QtCore import Qt, QPointF, QRectF


class ScalePointItem(QGraphicsItem):
    # TODO Enable Rectangle change by draging polygon points (blue circles)
    def __init__(self, scene, pos):
        super().__init__()

        self.setFlags(QGraphicsItem.ItemIsSelectable)
        self.setAcceptHoverEvents(True)
        self.local_rect = QRectF(-20, -20, 40, 40)
        self.x = pos[0]
        self.y = pos[1]
        self.r = 10
        self.setPos(QPointF(self.x, self.y))
        self.curr_scene = scene
        scene.clearSelection()
        self.color_green = QColor(0, 255, 0)
        self.color_blue = QColor(0, 0, 255)
        self.color_red = QColor(255, 0, 0)
        self.base_color = QColor("#55bcc2")
        self.color = self.base_color

    def handle_hover_event(self, QGraphicsSceneHoverEvent):
        modifiers = QApplication.keyboardModifiers()
        if self.curr_scene.picked_tool == 3:
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
        self.curr_scene.annotation_manager.scale_points[self.curr_scene.image_name].remove((self.x, self.y))
        self.curr_scene.removeItem(self)

    def mousePressEvent(self, QMouseEvent):
        # if left mouse button
        modifiers = QApplication.keyboardModifiers()
        if self.curr_scene.picked_tool == 3:
            if (QMouseEvent.button() == 1 and modifiers == Qt.ControlModifier):
                self.handle_removal_event()

    def boundingRect(self):
        return self.local_rect

    def paint(self, painter, option, widget):
        brush = QBrush(self.color)
        pen = QPen(brush, 5)
        painter.setPen(pen)
        painter.setBrush(brush)
        painter.setFont(QFont('Decorative', 10))
        painter.drawEllipse(QPointF(0, 0), 15, 15)
