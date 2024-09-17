import type { NextPage } from "next";
import Head from "next/head";
import Link from "next/link";

const Home: NextPage = () => {
  return (
    <div>
      <Head>
        <title>ARC Editor</title>
        <meta name="description" content="Edit and render ARC puzzles" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <main>
        <h1>ARC Editor</h1>
        <Link href="/task/ecaa0ec1">View Task ecaa0ec1</Link>
      </main>
    </div>
  );
};

export default Home;
