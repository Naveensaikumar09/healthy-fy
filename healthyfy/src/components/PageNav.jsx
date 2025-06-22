import { NavLink } from "react-router-dom"
import "./PageNav.css"
import Logo from "./Logo"
function PageNav({ispredictPage}) {
    return (
        
        <nav className="nav">
            <Logo  ispredictPage={ispredictPage} />
            <ul className="li-container">
            <li>
                <NavLink className="pageNavFields" to='/about' >About</NavLink>
            </li>
            <li>
                <NavLink className="pageNavFields" to='/predict' >Predict</NavLink>
            </li>
        </ul>
        </nav>
    )
}

export default PageNav
