import "./PredictPage.css";  
import PageNav from "../components/PageNav";  
import { useState } from "react";  
import { useNavigate } from "react-router-dom"; // Import useNavigate  

function PredictPage({ setResponseData }) { // Destructure props to get setResponseData  
    const [textmessage, setTextmessage] = useState("");  
    const [isLoading, setIsLoading] = useState(false);  
    const [error, setError] = useState(null);  

    const navigate = useNavigate(); // Initialize useNavigate  

    async function handlePredict() {  
        if (textmessage === "") return; // Prevent empty submissions  

        setIsLoading(true); // Set loading state  

        try {  
            const response = await fetch(`${import.meta.env.VITE_BACKEND_URL}/predict`, {  
                method: "POST",  
                headers: {  
                    "Content-Type": "application/json",  
                },  
                body: JSON.stringify({ paragraph: textmessage }), // Send the textmessage as 'paragraph'  
            });  

            if (!response.ok) {  
                throw new Error("Network response was not ok");  
            }  

            const data = await response.json();  
            setResponseData(data); // Update responseData in the parent component  
            setTextmessage(""); // Clear input field  

            // Redirect to the result page after receiving a valid response  
            navigate("/result"); // Redirect to the result page  

        } catch (error) {  
            console.error("Error:", error);  
            setError(error.message); // Set error message  
        } finally {  
            setIsLoading(false); // Reset loading state  
        }  
    }  

    return (  
        <div className="container">  
            <PageNav ispredictPage={true} />  
            <div className="left-panel">  
                <h1>  
                    We<br />  
                    predict<br />  
                    Your<br />  
                    Mental<br />  
                    Health  
                </h1>  
            </div>  
            <div className="right-panel">  
                <textarea  
                    value={textmessage}  
                    onChange={(e) => { setTextmessage(e.target.value); }}  
                    placeholder="Type something here..."  
                />  
                <button onClick={handlePredict} className="predict-button" disabled={isLoading}>  
                    {isLoading ? "Loading..." : "Predict"}  
                </button>  

                {error && <p className="error-message">{error}</p>} {/* Display error message */}  
            </div>  
        </div>  
    );  
}  

export default PredictPage;  