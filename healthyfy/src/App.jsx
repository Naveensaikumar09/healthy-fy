import { BrowserRouter,Routes,Route } from "react-router-dom"
import HomePage from './pages/HomePage'
import PredictPage from "./pages/PredictPage"
import ResultPage from "./pages/ResultPage"
import AboutPage from "./pages/AboutPage"
import { useState } from "react"
function App() {
  const [responseData,setResponseData] = useState({});
  return (
    <BrowserRouter>
       <Routes>
          <Route path='/' element={<HomePage/>} />
          <Route path='/predict' element={<PredictPage setResponseData={setResponseData}/>} />
          <Route path='/result' element={<ResultPage responseData={responseData}/>} />
          <Route path='/about' element={<AboutPage/>} />
       </Routes>
    </BrowserRouter>
  )
}

export default App
