import { Link } from "react-router-dom"
import PageNav from "../components/PageNav"
import "./HomePage.css"
function HomePage() {
    return (
    <main className="homepage">
      <div className="overlay"></div>
      <div className="content">
      <PageNav />
      <section >
        <Link to='/predict' className='StartPredicting'>Start predicting now</Link>
      </section>
      </div>
      
    </main>
    )
}

export default HomePage
