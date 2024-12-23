import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, Float, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.engine import create_engine
from sqlalchemy.orm import sessionmaker

from sqlalchemy import Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base

# Load data
df = pd.read_csv("/home/am/Documents/Software Development/10_Academy Training/week2/Data/Copy of Week2_challenge_data_source(CSV).csv")


from scripts.user_satisfaction_analysis import UserSatisfactionAnalyzer


# Load environment variables from .env file
# load_dotenv()

# Define the base class for our SQLAlchemy model
Base = declarative_base()

class UserSatisfaction(Base):
    __tablename__ = 'user_satisfaction'

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, nullable=False)  # For MSISDN/Number
    engagement_score = Column(Float, nullable=False)
    experience_score = Column(Float, nullable=False)
    satisfaction_score = Column(Float, nullable=False)
    # cluster = Column(String, nullable=False)

    def __init__(self, user_id, engagement_score, experience_score, satisfaction_score):
        self.user_id = user_id
        self.engagement_score = engagement_score
        self.experience_score = experience_score
        self.satisfaction_score = satisfaction_score
        # self.cluster = cluster
        self.user_satisfy = UserSatisfactionAnalyzer(df)

        # Load environment variables from .env file
        load_dotenv()

        # Initialize database connection parameters
        self.db_host = os.getenv("DB_HOST")
        self.db_port = os.getenv("DB_PORT")
        self.db_name = os.getenv("DB_NAME")
        self.db_user = os.getenv("DB_USER")
        self.db_password = os.getenv("DB_PASSWORD")

        # Construct the database URL
        self.db_url = f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"


    def export_to_postgresql(self):
      # Set up the database engine
      engine = create_engine(self.db_url)
      # Create a session factory
      Session = sessionmaker(bind=engine)

      session = Session()

      # Compute engagement and experience scores if not already computed
      if self.user_satisfy.engagement_scores is None:
          self.user_satisfy.assign_engagement_score()
      if self.user_satisfy.experience_scores is None:
          self.user_satisfy.assign_experience_score()

      # Now analyze user satisfaction
      merged_data, satisfaction_df, _ = self.user_satisfy.analyze_user_satisfaction()

      # Insert data into the database
      for _, row in satisfaction_df.iterrows():
          user_satisfaction_entry = UserSatisfaction(
              user_id =row["MSISDN/Number"],
              engagement_score=row['engagement_score'],
              experience_score=row['experience_score'],
              satisfaction_score=row['satisfaction_score']
          )
          session.add(user_satisfaction_entry)
      session.commit()






