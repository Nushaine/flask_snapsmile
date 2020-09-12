from flask_sqlalchemy import SQLAlchemy
from main import db
import sys
sys.path.insert(0, '/startup-slang/')


class UpperTeeth(db.Model):
    __tablename__ = 'upper teeth'
    id = db.Column(db.Integer, primary_key=True)
    image_name = db.Column(db.String(255))
    image = db.Column(db.BLOB)
    bbox_coords_id = db.Column(db.Integer, db.ForeignKey("upper_teeth_coords.id"))
    confidence_scores_id = db.Column(db.Integer, db.ForeignKey("upper_teeth_scores.id"))

    def __repr__(self):
      return f'{self.__tablename__}: image_name: {self.image_name}, image: {self.image}'