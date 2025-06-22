import React from 'react';  
import { Typography } from 'antd';  
import './ResultPage.css';  
import PageNav from '../components/PageNav';  

const { Title, Text } = Typography;  

const Result = ({ responseData }) => {  
  const traitData = [  
    { trait: 'Openness', value: responseData.Openness },  
    { trait: 'Conscientiousness', value: responseData.Conscientiousness },  
    { trait: 'Extraversion', value: responseData.Extraversion },  
    { trait: 'Agreeableness', value: responseData.Agreeableness },  
    { trait: 'Neuroticism', value: responseData.Neuroticism },  
    { trait: 'Risk', value: responseData.Risk }  
  ];  

  const colors = {  
    Openness: '#66E2DC',  
    Conscientiousness: '#4DB9E8',  
    Extraversion: '#2782C5',  
    Agreeableness: '#275A9C',  
    Neuroticism: '#A347B3',  
    Risk: '#FF0033'  
  };  

  const renderTraitCards = () => {  
    return traitData.map((trait) => (  
      <div key={trait.trait} className="trait-card">  
        <div className="trait-badge" style={{ backgroundColor: colors[trait.trait] }}>  
          <Title level={3} className="trait-name">  
            {trait.trait}  
          </Title>  
          <div className="trait-value-container">  
            <Title level={2} className="trait-value">  
              {trait.value}  
            </Title>  
            {trait.trait !== 'Risk' && (  
              <Text className="trait-score">Score</Text>  
            )}  
          </div>  
        </div>  
      </div>  
    ));  
  };  

  return (  
    <div className="page">  
      <PageNav />  
      <div className="result-container">  
         
          {renderTraitCards()}  
         
      </div>  
    </div>  
  );  
};  

export default Result;  