from PyQt5.QtGui import QBrush, QPen, QColor, QPolygonF
from PyQt5.QtWidgets import QGraphicsPolygonItem
from PyQt5.QtCore import QPointF

import numpy as np
from cv2 import convexHull
from widgets.annotation_graphics_items import AnnotationItem, InProgressAnnotation

class AnnotationTool():

    def __init__(self,annotation_manager, scene):
        self.annotation_manager = annotation_manager
        self.scene = scene
        self.annotation_undo_stack = []
        self.progress_annotation = None

    def draw_convex_hull(self):
        if len(self.annotation_manager.annotations_rect[self.scene.image_name]) > 1:
            if self.scene.polygon is not None:
                self.scene.removeItem(self.scene.polygon)
                self.scene.polygon = None
            points = []
            for annot in self.annotation_manager.annotations_rect[self.scene.image_name]:
                x, y, w, h = annot
                x = int(x)
                y = int(y)
                w = int(w)
                h = int(h)

                points.append((x, y))
                points.append((x + w, y))
                points.append((x, y + h))
                points.append((x + w, y + h))

            brush = QBrush(QColor("#1E90FF"))
            pen = QPen(brush, 3)

            polygon = QPolygonF([QPointF(x[0, 0], x[0, 1]) for x in convexHull(np.array(points))])
            self.scene.polygon = QGraphicsPolygonItem(polygon)
            self.scene.polygon.setPen(pen)
            self.scene.addItem(self.scene.polygon)
        else:
            if self.scene.polygon is not None:
                self.scene.removeItem(self.scene.polygon)
                self.scene.polygon = None


    def add_annotation(self, item_pos, annotation_pos):
        x1, y1, x2, y2 = annotation_pos
        x1 = max(0, min(x1, self.scene.image.width()))
        y1 = max(0, min(y1, self.scene.image.height()))
        x2 = max(0, min(x2, self.scene.image.width()))
        y2 = max(0, min(y2, self.scene.image.height()))

        tmp_annot = AnnotationItem(item_pos, self.scene, [x1, y1, x2 - x1, y2 - y1],
                                   annotation_negative=(self.scene.picked_tool == 2))
        self.scene.addItem(tmp_annot)
        if self.scene.picked_tool == 0:
            self.scene.annotations.add(tmp_annot)
            self.annotation_manager.annotations_rect[self.scene.image_name].add((tmp_annot.x, tmp_annot.y, tmp_annot.w, tmp_annot.h))
        else:
            self.scene.negative_annotations.add(tmp_annot)
            self.annotation_manager.negative_annotations_rect[self.scene.image_name].add((tmp_annot.x, tmp_annot.y, tmp_annot.w, tmp_annot.h))

        self.annotation_manager.annotations_changed[self.scene.image_name] = True
        self.scene.count_label.setText("Number of annotations: " + str(len(self.scene.annotations)))
        self.saved = False
        self.draw_convex_hull()
        self.annotation_undo_stack.append((1, tmp_annot))
        self.scene.action_undo_stack.append(self.scene.picked_tool)

    def undo_annotation(self):
        if len(self.annotation_undo_stack) > 0:
            action_status, changed_annotation = self.annotation_undo_stack.pop()
            # if action is annotation
            if action_status == 1:
                changed_annotation.handle_removal_event(removal_from_undo=True)

            # if action is removal
            elif action_status == 0:
                self.scene.addItem(changed_annotation)
                self.scene.annotations.add(changed_annotation)
                self.annotation_manager.annotations_rect[self.scene.image_name].add(
                    (changed_annotation.x, changed_annotation.y, changed_annotation.w, changed_annotation.h))
            return True
        else:
            return False

    def remove_progress_annotation(self):
        if self.progress_annotation is not None and self.progress_annotation.scene() == self.scene:
            self.scene.removeItem(self.progress_annotation)
            self.progress_annotation = None


    def add_progress_annotation(self, item_pos, annotation_pos, rectangular=False):
        x1, y1, x2, y2 = annotation_pos

        if self.progress_annotation is not None and self.progress_annotation.scene() == self.scene:
            self.scene.removeItem(self.progress_annotation)

        self.progress_annotation = InProgressAnnotation(item_pos, self.scene, [x1, y1, x2 - x1, y2 - y1], rectangular=rectangular)
        self.scene.addItem(self.progress_annotation)
