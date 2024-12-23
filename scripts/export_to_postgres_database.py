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
df = pd.read_csv("""
                    /home/am/Documents/Software Development/10_Academy Training/week2/Data/
                    Copy of Week2_challenge_data_source(CSV).csv
                 """)


from user_satisfaction_analysis import UserSatisfactionAnalyzer


# Load environment variables from .env file
load_dotenv()

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


    def export_to_postgresql(self):
        """Exports the final table to a PostgreSQL database using SQLAlchemy"""
        # Create SQLAlchemy engine
        engine = create_engine(self.user_satisfy.db_url)

        # Create all tables in the database
        Base.metadata.create_all(engine)

        # Create a session
        Session = sessionmaker(bind=engine)
        session = Session()

        # Get the merged data and satisfaction scores from analyze_user_satisfaction

        merged_data, satisfaction_df, _ = self.user_satisfy.analyze_user_satisfaction()

        # Compute the satisfaction_score in export_to_postgresql
        merged_data['satisfaction_score'] = (
            merged_data['engagement_score'] + merged_data['experience_score']) / 2

        # Insert data into the database
        for _, row in merged_data.iterrows():
            user_satisfaction = UserSatisfaction(
                user_id=row['MSISDN/Number'],  # Assuming 'MSISDN/Number' is the correct column name
                engagement_score=row['engagement_score'],
                experience_score=row['experience_score'],
                satisfaction_score=row['satisfaction_score']
            )
            session.add(user_satisfaction)

        # Commit the transaction
        session.commit()

        # Close the session
        session.close()

        print("Data exported successfully to PostgreSQL database.")





