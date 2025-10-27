
import TypedText from "@/components/TypedText";

const projects = [
  {
    title: "Chess Engine",
    description:
      "Modern chess engine written in c++ using modern search techniques and NNUE for evaluation.",
    link: "/projects/chess-engine",
    image: "/chess_board.svg",
  },
  {
    title: "Neural Network Framework",
    description:
      "A minimal deep learning library built in Python with CUDA support.",
    link: "/projects/nn-framework",
    image: "/neural-network.svg",
  },
];

type ProjectCardProps = {
  title: string;
  description: string;
  link: string;
  image: string;
};

function ProjectCard({ title, description, link, image }: ProjectCardProps) {
  return (
    <a href={link}>
      <div className="rounded-lg shadow-md p-6 w-80 min-h-104 bg-(--surface) hover:-translate-y-2 transition-transform duration-200 border-2 border-(--surface-light)">
        <img
          src={image}
          className="w-full h-48 object-contain mb-4 rounded-md filter invert"
        />
        <h3 className="text-2xl font-semibold mb-2 text-white">{title}</h3>
        <p className="grow mb-4">{description}</p>
      </div>
    </a>
  );
}

export default function Home() {
  return (
    <main className="flex min-h-screen w-full flex-col">
      <section
        id="about"
        className="h-screen w-full flex flex-col items-center justify-center"
      >
        <p className="text-(--text-muted) text-lg mb-2">Hi, I'm</p>
        <h1 className="text-6xl font-bold text-(--text-strong)">
          Andreas Sæveraas
        </h1>
        <h2 className="text-3xl mt-2">
          And I'm a{" "}
          <TypedText
            className="text-(--accent)"
            strings={[
              "Backend Developer",
              "C++ Engineer",
              "AI Enthusiast",
              "Systems Programmer",
            ]}
          ></TypedText>
        </h2>
        <p className="mt-6 text-xl max-w-lg text-(--text-muted)">
          I build high-performance software — from chess engines to neural
          network frameworks.
        </p>
      </section>
      <section
        id="projects"
        className="h-screen w-full flex flex-col items-center justify-center"
      >
        <h2 className="text-4xl font-semibold mb-8">Projects</h2>

        <div className="grid gap-8 sm:grid-cols-2 lg:grid-cols-3 justify-items-center">
          {projects.map((project) => (
            <ProjectCard
              key={project.title}
              title={project.title}
              description={project.description}
              link={project.link}
              image={project.image}
            />
          ))}
        </div>
      </section>
      <section
        id="contact"
        className="h-screen w-full flex flex-col items-center justify-center"
      >
        <h2 className="text-4xl font-semibold mt-10">Contact</h2>
      </section>
    </main>
  );
}
