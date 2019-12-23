from PyQt5.QtCore import QPointF
from PyQt5.QtGui import QBrush, QColor, QPen
from PyQt5.QtWidgets import QGraphicsEllipseItem


class RemovalTool():
    def __init__(self,annotation_manager, scene):
        self.annotation_manager = annotation_manager
        self.scene = scene
        self.scene.removal_circle_radius = 100
        self.current_remove_action_set = set()
        self.remove_undo_stack = []

    def remove_action(self, pos):
        num_removed = 0
        removable_annotations = filter(
            lambda annot: ((annot.x - pos.x()) ** 2 + (annot.y - pos.y()) ** 2) ** 0.5 < self.scene.removal_circle_radius / 2
                          or ((annot.x + annot.w - pos.x()) ** 2 + (
                        annot.y + annot.h - pos.y()) ** 2) ** 0.5 < self.scene.removal_circle_radius / 2
                          or ((annot.x + annot.w - pos.x()) ** 2 + (
                        annot.y - pos.y()) ** 2) ** 0.5 < self.scene.removal_circle_radius / 2
                          or ((annot.x - pos.x()) ** 2 + (
                        annot.y + annot.h - pos.y()) ** 2) ** 0.5 < self.scene.removal_circle_radius / 2
            , list(self.scene.annotations))

        removable_annotations_neg = filter(
            lambda annot: ((annot.x - pos.x()) ** 2 + (
                        annot.y - pos.y()) ** 2) ** 0.5 < self.scene.removal_circle_radius / 2
                          or ((annot.x + annot.w - pos.x()) ** 2 + (
                    annot.y + annot.h - pos.y()) ** 2) ** 0.5 < self.scene.removal_circle_radius / 2
                          or ((annot.x + annot.w - pos.x()) ** 2 + (
                    annot.y - pos.y()) ** 2) ** 0.5 < self.scene.removal_circle_radius / 2
                          or ((annot.x - pos.x()) ** 2 + (
                    annot.y + annot.h - pos.y()) ** 2) ** 0.5 < self.scene.removal_circle_radius / 2
            , list(self.scene.negative_annotations))

        for annot in removable_annotations_neg:
            num_removed += 1
            self.annotation_manager.negative_annotations_rect[self.scene.image_name].remove((annot.x, annot.y, annot.w, annot.h))
            self.scene.negative_annotations.remove(annot)
            self.scene.removeItem(annot)
            self.current_remove_action_set.add(annot)

        for annot in removable_annotations:
            num_removed += 1
            self.annotation_manager.annotations_rect[self.scene.image_name].remove((annot.x, annot.y, annot.w, annot.h))
            self.scene.annotations.remove(annot)
            self.scene.removeItem(annot)
            self.current_remove_action_set.add(annot)

        return num_removed

    def undo_remove_action(self):
        if len(self.remove_undo_stack) > 0:
            previously_removed_set = self.remove_undo_stack.pop()
            for tmp_annot in previously_removed_set:
                self.scene.addItem(tmp_annot)
                self.scene.annotations.add(tmp_annot)
                self.annotation_manager.annotations_rect[self.scene.image_name].add((tmp_annot.x, tmp_annot.y, tmp_annot.w, tmp_annot.h))
            return True
        else:
            return False

    def end_of_action(self):
        if len(self.current_remove_action_set) > 0:
            self.remove_undo_stack.append(set(self.current_remove_action_set))
            self.current_remove_action_set = set()
            if len(self.remove_undo_stack) > 50:
                self.remove_undo_stack.pop(0)

            return True
        else:
            return False

    def draw_removal_circle(self, t_pos):
        pos = QPointF(min(self.scene.width(), max(0, t_pos.x())), min(self.scene.height(), max(0, t_pos.y())))
        radius = self.scene.removal_circle_radius
        brush = QBrush(QColor(180, 0, 0))
        pen = QPen(brush, 3)
        if self.scene.removal_circle_item is not None:
            self.scene.removeItem(self.scene.removal_circle_item)
        self.scene.removal_circle_item = QGraphicsEllipseItem(pos.x() - radius // 2, pos.y() - radius // 2, radius, radius)
        self.scene.removal_circle_item.setPen(pen)
        self.scene.addItem(self.scene.removal_circle_item)

