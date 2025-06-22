import { Link } from "react-router-dom";
import logo from "../assets/logo1.jpg"
import "./Logo.css"

function Logo({ispredictPage}) {
  return <Link className="logo-contianer" to='/'>
    
    <div className="logo-contianer-container">
    <img src={logo} alt="healthify logo" className='logo' />
    </div>
    {!ispredictPage&&<span>Healthify</span>}
   
  </Link>;
}

export default Logo;