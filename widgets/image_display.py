from PyQt5.QtWidgets import QApplication, QGraphicsView, QGraphicsScene, \
    QGraphicsPolygonItem, QGraphicsLineItem
from PyQt5.QtGui import QColor, QFont, QCursor, QImage, QPixmap, QPolygonF, QPen, QBrush
from PyQt5.QtCore import Qt, QPointF, QLineF
import numpy as np
from utils.removal_tool import RemovalTool
from utils.annotation_tool import AnnotationTool
from .polygon_point import PolygonPointItem
from .scale_point import ScalePointItem
from .annotation_graphics_items import AnnotationItem

class GraphicsScene(QGraphicsScene):
    def __init__(self, annotation_manager, count_label, poly_label, density_label, parent=None):
        super(GraphicsScene, self).__init__(parent)
        self.annotations = set()
        self.negative_annotations = set()
        self.roi_point_items = set()
        self.scale_point_items = []
        self.polygon_surface = 0.0


        self.removal_tool = RemovalTool(annotation_manager, self)
        self.annotation_tool = AnnotationTool(annotation_manager, self)

        self.currently_drawing = False
        self.currently_deleting = False
        self.image_name = None
        self.image_path = None
        self.curr_image = None
        self.annotation_manager = annotation_manager
        self.count_label = count_label
        self.poly_label = poly_label
        self.density_label = density_label
        self.saved = True
        self.enabled = False

        self.polygon = None
        self.roi_polygon = None
        self.scale_line = None
        self.picked_tool = 0
        self.removal_circle_item = None
        self.removal_circle_radius = 100

        # stack of tuples such as: (add/remove, annotation_item)
        self.annotation_undo_stack = []

        # stack of ints depicting which action needs undoing
        self.action_undo_stack = []

    def add_image(self, image_path):
        image = QImage(image_path)
        self.image = image
        self.original_image = image.copy()
        h, w = image.height(), image.width()
        self.pixmap_item = self.addPixmap(QPixmap.fromImage(image))

    def change_annotations_on_image(self):
        print("Changing image annotations")
        self.enabled = True
        self.clear()
        self.annotations = set()
        self.negative_annotations = set()

        self.polygon = None
        self.roi_polygon = None
        self.scale_line = None
        self.removal_circle_item = None
        self.annotation_undo_stack = []
        self.action_undo_stack = []

        self.add_image(self.image_path)
        self.setSceneRect(self.itemsBoundingRect())


        for a in self.annotation_manager.annotations_rect[self.image_name]:
            tmp_annot = AnnotationItem(QPointF(a[0], a[1]), self, a)
            self.annotations.add(tmp_annot)
            self.addItem(tmp_annot)

        for a in self.annotation_manager.negative_annotations_rect[self.image_name]:
            tmp_annot = AnnotationItem(QPointF(a[0], a[1]), self, a, annotation_negative=True)
            self.negative_annotations.add(tmp_annot)
            self.addItem(tmp_annot)

        for a in self.annotation_manager.roi_points[self.image_name]:
            tmp_annot = PolygonPointItem(self, (a[0], a[1]))
            self.roi_point_items.add(tmp_annot)
            self.addItem(tmp_annot)

        for a in self.annotation_manager.scale_points[self.image_name]:
            tmp_annot = ScalePointItem(self, (a[0], a[1]))
            self.scale_point_items.append(tmp_annot)
            self.addItem(tmp_annot)


        self.annotation_tool.draw_convex_hull()
        self.draw_scale_line()
        self.draw_roi_polygon()
        self.count_label.setText("Number of annotations: " + str(len(self.annotations)))

        self.update()

    def change_image(self, image_path, image_name):
        self.enabled = True
        self.clear()
        self.annotations = set()
        self.negative_annotations = set()
        self.roi_point_items = set()
        self.scale_point_items = []

        self.polygon = None
        self.roi_polygon = None
        self.scale_line = None
        self.removal_circle_item = None
        self.annotation_undo_stack = []
        self.action_undo_stack = []

        self.add_image(image_path)
        self.setSceneRect(self.itemsBoundingRect())

        self.image_path = image_path
        self.annotation_manager.image_shapes[image_name] = [self.image.width(), self.image.height()]


        for a in self.annotation_manager.annotations_rect[image_name]:
            tmp_annot = AnnotationItem(QPointF(a[0], a[1]), self, a)
            self.annotations.add(tmp_annot)
            self.addItem(tmp_annot)

        for a in self.annotation_manager.negative_annotations_rect[image_name]:
            tmp_annot = AnnotationItem(QPointF(a[0], a[1]), self, a, annotation_negative=True)
            self.negative_annotations.add(tmp_annot)
            self.addItem(tmp_annot)

        for a in self.annotation_manager.roi_points[image_name]:
            tmp_annot = PolygonPointItem(self, (a[0], a[1]))
            self.roi_point_items.add(tmp_annot)
            self.addItem(tmp_annot)

        for a in self.annotation_manager.scale_points[image_name]:
            tmp_annot = ScalePointItem(self, (a[0], a[1]))
            self.scale_point_items.append(tmp_annot)
            self.addItem(tmp_annot)


        self.image_name = image_name
        self.count_label.setText("Number of annotations: " + str(len(self.annotations)))
        self.annotation_tool.draw_convex_hull()
        self.draw_roi_polygon()
        self.draw_scale_line()
        self.update()


    def get_annotations(self):
        return self.annotations

    def get_polygon_surface(self, x, y):
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    def draw_roi_polygon(self):
        if len(self.annotation_manager.roi_points[self.image_name]) > 2:
            if self.roi_polygon != None:
                self.removeItem(self.roi_polygon)
                self.roi_polygon = None
            brush = QBrush(QColor("#fc8803"))
            pen = QPen(brush, 5)

            polygon = QPolygonF(
                [QPointF(x[0], x[1]) for x in np.array(self.annotation_manager.roi_points[self.image_name])])
            self.roi_polygon = QGraphicsPolygonItem(polygon)
            self.roi_polygon.setPen(pen)
            self.addItem(self.roi_polygon)

            polygon_points = np.array(self.annotation_manager.roi_points[self.image_name])
            polygon_surface = self.get_polygon_surface(polygon_points[:,0], polygon_points[:,1])
            self.polygon_surface = polygon_surface
            if len(self.annotation_manager.scale_points[self.image_name]) == 2:
                x1, y1 = self.annotation_manager.scale_points[self.image_name][0]
                x2, y2 = self.annotation_manager.scale_points[self.image_name][1]
                line_length = ((x1-x2)**2 + (y1-y2)**2)**0.5 * 0.2
                polygon_surface_cm = polygon_surface / (line_length**2)
                self.poly_label.setText("Surface of polygon (cm): " + str(polygon_surface_cm))
                self.polygon_surface = polygon_surface_cm
            else:
                self.poly_label.setText("Surface of polygon (pixels): " + str(polygon_surface))
            density = np.round(len(self.annotations) / (self.polygon_surface+1e-16),6)
            self.density_label.setText("Object density: " + str(density))



        else:
            if self.roi_polygon != None:
                self.removeItem(self.roi_polygon)
                self.roi_polygon = None
            self.density_label.setText("Object density: None")

    def draw_scale_line(self):
        if len(self.annotation_manager.scale_points[self.image_name]) == 2:
            if self.scale_line != None:
                self.removeItem(self.scale_line)
                self.scale_line = None
            brush = QBrush(QColor("#55bcc2"))
            pen = QPen(brush, 7)

            x1,y1 = self.annotation_manager.scale_points[self.image_name][0]
            x2,y2 = self.annotation_manager.scale_points[self.image_name][1]

            sc_line = QLineF(x1,y1,x2,y2)
            self.scale_line = QGraphicsLineItem(sc_line)
            self.scale_line.setPen(pen)
            self.addItem(self.scale_line)
            if len(self.annotation_manager.roi_points[self.image_name]) > 2:
                polygon_points = np.array(self.annotation_manager.roi_points[self.image_name])
                polygon_surface = self.get_polygon_surface(polygon_points[:,0], polygon_points[:,1])
                line_length = ((x1-x2)**2 + (y1-y2)**2)**0.5 * 0.2
                polygon_surface_cm = np.round(polygon_surface / (line_length**2),3)
                self.poly_label.setText("Surface of polygon (cm): " + str(polygon_surface_cm))
                self.polygon_surface = polygon_surface_cm
                density = np.round(len(self.annotations) / (self.polygon_surface+1e-16),6)
                self.density_label.setText("Object density: " + str(density))


        else:
            if self.scale_line != None:
                self.removeItem(self.scale_line)
                self.scale_line = None


    def keyPressEvent(self, QKeyEvent):
        # Changes the cursor icon when deleting
        if self.picked_tool == 0 or self.picked_tool == 2:
            if QKeyEvent.key() == Qt.Key_Control and not self.currently_deleting:
                self.currently_deleting = True
                QApplication.setOverrideCursor(QCursor(Qt.CrossCursor))

    def keyReleaseEvent(self, QKeyEvent):
        # Changes the cursor icon back to normal when deleting stops
        if self.picked_tool == 0 or self.picked_tool == 2:
            if QKeyEvent.key() == Qt.Key_Control and self.currently_deleting:
                self.currently_deleting = False
                QApplication.setOverrideCursor(QCursor(Qt.ArrowCursor))



    def handle_remove_action(self, pos):
        num_removed = self.removal_tool.remove_action(pos)
        if num_removed > 0:
            self.annotation_manager.annotations_changed[self.image_name] = True
            self.count_label.setText("Number of annotations: " + str(len(self.annotations)))
            if self.polygon_surface > 0.0:
                density = np.round(len(self.annotations) / (self.polygon_surface+1e-16),6)
                self.density_label.setText("Object density: " + str(density))
            self.annotation_tool.draw_convex_hull()
            self.saved = False

    def handle_undo_remove_action(self):
        removed = self.removal_tool.undo_remove_action()
        if removed:
            self.annotation_manager.annotations_changed[self.image_name] = True
            self.count_label.setText("Number of annotations: " + str(len(self.annotations)))
            if self.polygon_surface > 0.0:
                density = np.round(len(self.annotations) / (self.polygon_surface+1e-16),6)
                self.density_label.setText("Object density: " + str(density))
            self.annotation_tool.draw_convex_hull()
            self.saved = False

    def handle_undo_annotation_action(self):
        removed = self.annotation_tool.undo_annotation()
        if removed:
            self.annotation_manager.annotations_changed[self.image_name] = True
            self.count_label.setText("Number of annotations: " + str(len(self.annotations)))
            if self.polygon_surface > 0.0:
                density = np.round(len(self.annotations) / (self.polygon_surface+1e-16),6)
                self.density_label.setText("Object density: " + str(density))
            self.saved = False
            self.annotation_tool.draw_convex_hull()

    def handle_undo_action(self):
        print("Test")
        if len(self.action_undo_stack) > 0:
            tool_used = self.action_undo_stack.pop()
            if tool_used == 0 or tool_used==2:
                print("Undo for annotations")
                self.handle_undo_annotation_action()
            elif tool_used == 1:
                print("Undo remove")
                self.handle_undo_remove_action()

    def mousePressEvent(self, QMouseEvent):
        super(GraphicsScene, self).mousePressEvent(QMouseEvent)
        # if left mouse button
        pos = QMouseEvent.scenePos()
        modifiers = QApplication.keyboardModifiers()
        # if tool is annotation adding
        if self.picked_tool == 0 or self.picked_tool == 2:
            if (QMouseEvent.button() == 1):
                self.currently_drawing = True
                self.annotation_tool.add_progress_annotation(pos, [pos.x(), pos.y(), pos.x() + 1, pos.y() + 1])
                self.start_x = pos.x()
                self.start_y = pos.y()
        elif self.picked_tool == 1 and QMouseEvent.buttons() == Qt.LeftButton:
            self.handle_remove_action(pos)
        elif self.picked_tool == 3 and QMouseEvent.button() == 1 and modifiers != Qt.ControlModifier:
            print("poly tool pressed")
            tmp_annot = PolygonPointItem(self, (pos.x(),pos.y()))
            self.currently_drawing = False
            self.addItem(tmp_annot)
            self.roi_point_items.add(tmp_annot)
            self.annotation_manager.roi_points[self.image_name].append((pos.x(),pos.y()))
            self.draw_roi_polygon()
        elif self.picked_tool == 4 and QMouseEvent.button() == 1 and modifiers != Qt.ControlModifier:
            print("scale tool pressed")
            tmp_annot = ScalePointItem(self, (pos.x(),pos.y()))
            self.currently_drawing = False
            self.addItem(tmp_annot)
            self.scale_point_items.append(tmp_annot)
            if len(self.scale_point_items) == 3:
                self.scale_point_items[0].handle_removal_event()
                self.scale_point_items = self.scale_point_items[1:]
            self.annotation_manager.scale_points[self.image_name].append((pos.x(),pos.y()))
            self.draw_scale_line()

        self.update()

    def mouseMoveEvent(self, QMouseEvent):
        # pos = QPointF(QMouseEvent.pos())
        super(GraphicsScene, self).mouseMoveEvent(QMouseEvent)
        pos = QMouseEvent.scenePos()
        modifiers = QApplication.keyboardModifiers()

        # if tool is annotation adding
        if self.picked_tool == 0 or self.picked_tool == 2 or self.picked_tool == 3:
            if self.currently_drawing:
                self.end_x = pos.x()
                self.end_y = pos.y()
                x1 = self.end_x
                y1 = self.end_y
                x2 = self.start_x
                y2 = self.start_y
                if self.picked_tool == 3 or self.picked_tool == 2:
                    self.annotation_tool.add_progress_annotation(pos, [x1, y1, x2, y2], rectangular=True)
                else:
                    self.annotation_tool.add_progress_annotation(pos, [x1, y1, x2, y2])

        elif self.picked_tool == 1 and QMouseEvent.buttons() == Qt.LeftButton:
            self.handle_remove_action(pos)

        if self.picked_tool == 1 or self.picked_tool == 4:
            self.removal_tool.draw_removal_circle(pos)

        self.update()

    def mouseReleaseEvent(self, QMouseEvent):
        pos = QMouseEvent.scenePos()
        # Handle adding positive/negative anotations
        if self.picked_tool == 0 or self.picked_tool == 2:
            if (QMouseEvent.button() == 1):
                self.currently_drawing = False
                self.end_x = pos.x()
                self.end_y = pos.y()
                # add annotation
                if (abs(self.start_x - self.end_x) > 1 and abs(self.start_y - self.end_y) > 1):
                    self.annotation_tool.add_annotation(pos, [self.start_x, self.start_y, self.end_x, self.end_y])
                # remove in progress annotation
                self.annotation_tool.remove_progress_annotation()
                if self.polygon_surface > 0.0:
                    density = np.round(len(self.annotations) / (self.polygon_surface + 1e-16), 6)
                    self.density_label.setText("Object density: " + str(density))


        # Save remove action to undo stack
        elif self.picked_tool == 1 and QMouseEvent.button() == 1:
            removed = self.removal_tool.end_of_action()
            if removed:
                self.action_undo_stack.append(self.picked_tool)

        self.update()

    def drawRect(self, event, qp, rect):
        qp.setPen(QColor(168, 34, 3))
        qp.setFont(QFont('Decorative', 10))
        qp.drawRect(*rect)
        self.update()


class ImageDisplayView(QGraphicsView):
    def __init__(self, parent=None):
        super(ImageDisplayView, self).__init__(parent)

    def wheelEvent(self, event):
        """
        Zoom in or out of the view.
        """
        zoomInFactor = 1.25
        zoomOutFactor = 1 / zoomInFactor
        # Save the scene pos
        oldPos = self.mapToScene(event.pos())
        # Zoom
        if event.angleDelta().y() > 0:
            zoomFactor = zoomInFactor
        else:
            zoomFactor = zoomOutFactor
        self.scale(zoomFactor, zoomFactor)

        # Get the new position
        newPos = self.mapToScene(event.pos())
        # Move scene to old position
        delta = newPos - oldPos
        self.translate(delta.x(), delta.y())

        viewCenter = self.mapToScene(self.viewport().rect().center())
        self.centerOn((newPos + viewCenter) / 2)

    def keyPressEvent(self, QKeyEvent):
        if QKeyEvent.key() == Qt.Key_Right:
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() + 10)
        if QKeyEvent.key() == Qt.Key_Left:
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - 10)
        if QKeyEvent.key() == Qt.Key_Down:
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() + 10)
        if QKeyEvent.key() == Qt.Key_Up:
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - 10)
