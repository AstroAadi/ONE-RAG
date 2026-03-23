"""
books_api.py
─────────────────────────────────────────────────────────────────────
Books Library API with Bearer token auth and pagination.

Demo credentials:
    username: admin   password: admin123   token: admin-token-abc
    username: guest   password: guest123   token: guest-token-xyz

Run:
    pip install fastapi uvicorn
    uvicorn books_api:app --reload --port 8090

Example:
    curl -X POST "http://localhost:8090/auth/token" \
         -H "Content-Type: application/json" \
         -d '{"username":"admin","password":"admin123"}'

    curl "http://localhost:8090/api/v1/books?page=1&page_size=5" \
         -H "Authorization: Bearer admin-token-abc"
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Query, Depends, Security
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import uvicorn

# ─────────────────────────────────────────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────────────────────────────────────────

class PaginationMeta(BaseModel):
    page:        int
    page_size:   int
    total_items: int
    total_pages: int
    has_next:    bool
    has_prev:    bool
    next_page:   Optional[int] = None
    prev_page:   Optional[int] = None


class Author(BaseModel):
    name:        str
    nationality: str
    birth_year:  int


class Publisher(BaseModel):
    name:    str
    country: str


class Book(BaseModel):
    book_id:       str
    title:         str
    isbn:          str
    genre:         str
    language:      str
    pages:         int
    published_year: int
    rating:        float
    available:     bool
    author:        Author
    publisher:     Publisher
    tags:          List[str]


class PaginatedBooks(BaseModel):
    data:       List[Book]
    pagination: PaginationMeta
    filters:    Dict[str, Any] = Field(default_factory=dict)


class TokenRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"
    role:         str


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

_BOOKS: List[Dict] = [
    # ── Technology (bk-001 to bk-020) ────────────────────────────────────────
    {"book_id":"bk-001","title":"The Pragmatic Programmer","isbn":"978-0-13-595705-9","genre":"Technology","language":"English","pages":352,"published_year":2019,"rating":4.8,"available":True,"author":{"name":"David Thomas","nationality":"British","birth_year":1956},"publisher":{"name":"Addison-Wesley","country":"USA"},"tags":["programming","software-engineering","best-practices"]},
    {"book_id":"bk-002","title":"Clean Code","isbn":"978-0-13-235088-4","genre":"Technology","language":"English","pages":464,"published_year":2008,"rating":4.7,"available":True,"author":{"name":"Robert C. Martin","nationality":"American","birth_year":1952},"publisher":{"name":"Prentice Hall","country":"USA"},"tags":["programming","refactoring","clean-code"]},
    {"book_id":"bk-003","title":"Designing Data-Intensive Applications","isbn":"978-1-4493-7332-0","genre":"Technology","language":"English","pages":616,"published_year":2017,"rating":4.9,"available":True,"author":{"name":"Martin Kleppmann","nationality":"German","birth_year":1984},"publisher":{"name":"O'Reilly","country":"USA"},"tags":["databases","distributed-systems","backend","architecture"]},
    {"book_id":"bk-004","title":"Python Crash Course","isbn":"978-1-59327-928-8","genre":"Technology","language":"English","pages":544,"published_year":2019,"rating":4.7,"available":True,"author":{"name":"Eric Matthes","nationality":"American","birth_year":1981},"publisher":{"name":"No Starch Press","country":"USA"},"tags":["python","programming","beginner","tutorial"]},
    {"book_id":"bk-005","title":"The Art of Computer Programming","isbn":"978-0-13-468599-1","genre":"Technology","language":"English","pages":3168,"published_year":1968,"rating":4.9,"available":False,"author":{"name":"Donald Knuth","nationality":"American","birth_year":1938},"publisher":{"name":"Addison-Wesley","country":"USA"},"tags":["algorithms","computer-science","programming","classic"]},
    {"book_id":"bk-006","title":"Introduction to Algorithms","isbn":"978-0-26-204630-5","genre":"Technology","language":"English","pages":1312,"published_year":2009,"rating":4.6,"available":True,"author":{"name":"Thomas H. Cormen","nationality":"American","birth_year":1956},"publisher":{"name":"MIT Press","country":"USA"},"tags":["algorithms","data-structures","computer-science","textbook"]},
    {"book_id":"bk-007","title":"Structure and Interpretation of Computer Programs","isbn":"978-0-26-251087-5","genre":"Technology","language":"English","pages":657,"published_year":1996,"rating":4.8,"available":True,"author":{"name":"Harold Abelson","nationality":"American","birth_year":1947},"publisher":{"name":"MIT Press","country":"USA"},"tags":["programming","lisp","computer-science","classic"]},
    {"book_id":"bk-008","title":"The Mythical Man-Month","isbn":"978-0-20-183595-3","genre":"Technology","language":"English","pages":336,"published_year":1995,"rating":4.5,"available":True,"author":{"name":"Frederick P. Brooks Jr.","nationality":"American","birth_year":1931},"publisher":{"name":"Addison-Wesley","country":"USA"},"tags":["software-engineering","project-management","classic","teams"]},
    {"book_id":"bk-009","title":"Refactoring","isbn":"978-0-13-468599-5","genre":"Technology","language":"English","pages":448,"published_year":2018,"rating":4.6,"available":False,"author":{"name":"Martin Fowler","nationality":"British","birth_year":1963},"publisher":{"name":"Addison-Wesley","country":"USA"},"tags":["refactoring","clean-code","design-patterns","programming"]},
    {"book_id":"bk-010","title":"Domain-Driven Design","isbn":"978-0-32-112521-7","genre":"Technology","language":"English","pages":560,"published_year":2003,"rating":4.5,"available":True,"author":{"name":"Eric Evans","nationality":"American","birth_year":1960},"publisher":{"name":"Addison-Wesley","country":"USA"},"tags":["ddd","architecture","software-design","patterns"]},
    {"book_id":"bk-011","title":"Working Effectively with Legacy Code","isbn":"978-0-13-117705-5","genre":"Technology","language":"English","pages":456,"published_year":2004,"rating":4.4,"available":True,"author":{"name":"Michael C. Feathers","nationality":"American","birth_year":1967},"publisher":{"name":"Prentice Hall","country":"USA"},"tags":["legacy-code","refactoring","testing","programming"]},
    {"book_id":"bk-012","title":"Kubernetes in Action","isbn":"978-1-61-729735-0","genre":"Technology","language":"English","pages":624,"published_year":2017,"rating":4.7,"available":True,"author":{"name":"Marko Luksa","nationality":"Slovenian","birth_year":1979},"publisher":{"name":"Manning","country":"USA"},"tags":["kubernetes","devops","containers","cloud"]},
    {"book_id":"bk-013","title":"Docker Deep Dive","isbn":"978-1-52-110338-9","genre":"Technology","language":"English","pages":272,"published_year":2023,"rating":4.5,"available":True,"author":{"name":"Nigel Poulton","nationality":"British","birth_year":1975},"publisher":{"name":"Self-Published","country":"UK"},"tags":["docker","containers","devops","cloud"]},
    {"book_id":"bk-014","title":"Clean Architecture","isbn":"978-0-13-468599-9","genre":"Technology","language":"English","pages":432,"published_year":2017,"rating":4.5,"available":False,"author":{"name":"Robert C. Martin","nationality":"American","birth_year":1952},"publisher":{"name":"Prentice Hall","country":"USA"},"tags":["architecture","software-design","clean-code","patterns"]},
    {"book_id":"bk-015","title":"Fluent Python","isbn":"978-1-49-192063-4","genre":"Technology","language":"English","pages":1012,"published_year":2022,"rating":4.8,"available":True,"author":{"name":"Luciano Ramalho","nationality":"Brazilian","birth_year":1968},"publisher":{"name":"O'Reilly","country":"USA"},"tags":["python","advanced","programming","idiomatic"]},
    {"book_id":"bk-016","title":"The Phoenix Project","isbn":"978-1-94-278028-1","genre":"Technology","language":"English","pages":432,"published_year":2013,"rating":4.6,"available":True,"author":{"name":"Gene Kim","nationality":"American","birth_year":1971},"publisher":{"name":"IT Revolution Press","country":"USA"},"tags":["devops","business","novel","it-management"]},
    {"book_id":"bk-017","title":"Site Reliability Engineering","isbn":"978-1-49-192912-4","genre":"Technology","language":"English","pages":552,"published_year":2016,"rating":4.7,"available":True,"author":{"name":"Niall Richard Murphy","nationality":"Irish","birth_year":1978},"publisher":{"name":"O'Reilly","country":"USA"},"tags":["sre","devops","reliability","google"]},
    {"book_id":"bk-018","title":"Building Microservices","isbn":"978-1-49-233401-7","genre":"Technology","language":"English","pages":616,"published_year":2021,"rating":4.6,"available":False,"author":{"name":"Sam Newman","nationality":"British","birth_year":1978},"publisher":{"name":"O'Reilly","country":"USA"},"tags":["microservices","architecture","distributed-systems","api"]},
    {"book_id":"bk-019","title":"JavaScript: The Good Parts","isbn":"978-0-59-651774-8","genre":"Technology","language":"English","pages":176,"published_year":2008,"rating":4.4,"available":True,"author":{"name":"Douglas Crockford","nationality":"American","birth_year":1955},"publisher":{"name":"O'Reilly","country":"USA"},"tags":["javascript","programming","web","language"]},
    {"book_id":"bk-020","title":"You Don't Know JS","isbn":"978-1-49-190552-5","genre":"Technology","language":"English","pages":296,"published_year":2015,"rating":4.7,"available":True,"author":{"name":"Kyle Simpson","nationality":"American","birth_year":1980},"publisher":{"name":"O'Reilly","country":"USA"},"tags":["javascript","advanced","programming","web"]},

    # ── Fiction & Classics (bk-021 to bk-040) ────────────────────────────────
    {"book_id":"bk-021","title":"1984","isbn":"978-0-45-152493-5","genre":"Fiction","language":"English","pages":328,"published_year":1949,"rating":4.7,"available":True,"author":{"name":"George Orwell","nationality":"British","birth_year":1903},"publisher":{"name":"Secker & Warburg","country":"UK"},"tags":["dystopia","classic","politics","surveillance"]},
    {"book_id":"bk-022","title":"The Great Gatsby","isbn":"978-0-74-327356-5","genre":"Fiction","language":"English","pages":180,"published_year":1925,"rating":4.1,"available":True,"author":{"name":"F. Scott Fitzgerald","nationality":"American","birth_year":1896},"publisher":{"name":"Scribner","country":"USA"},"tags":["classic","american-literature","tragedy"]},
    {"book_id":"bk-023","title":"To Kill a Mockingbird","isbn":"978-0-06-112008-4","genre":"Fiction","language":"English","pages":336,"published_year":1960,"rating":4.8,"available":True,"author":{"name":"Harper Lee","nationality":"American","birth_year":1926},"publisher":{"name":"J.B. Lippincott","country":"USA"},"tags":["classic","race","justice","american-south"]},
    {"book_id":"bk-024","title":"Pride and Prejudice","isbn":"978-0-14-143951-8","genre":"Fiction","language":"English","pages":432,"published_year":1813,"rating":4.6,"available":True,"author":{"name":"Jane Austen","nationality":"British","birth_year":1775},"publisher":{"name":"Penguin Classics","country":"UK"},"tags":["classic","romance","british-literature","satire"]},
    {"book_id":"bk-025","title":"Crime and Punishment","isbn":"978-0-14-044913-6","genre":"Fiction","language":"English","pages":671,"published_year":1866,"rating":4.5,"available":False,"author":{"name":"Fyodor Dostoevsky","nationality":"Russian","birth_year":1821},"publisher":{"name":"Penguin Classics","country":"UK"},"tags":["classic","russian-literature","psychology","philosophy"]},
    {"book_id":"bk-026","title":"The Catcher in the Rye","isbn":"978-0-31-676948-0","genre":"Fiction","language":"English","pages":277,"published_year":1951,"rating":4.0,"available":True,"author":{"name":"J.D. Salinger","nationality":"American","birth_year":1919},"publisher":{"name":"Little, Brown","country":"USA"},"tags":["classic","coming-of-age","american-literature","adolescence"]},
    {"book_id":"bk-027","title":"Brave New World","isbn":"978-0-06-085052-4","genre":"Fiction","language":"English","pages":311,"published_year":1932,"rating":4.4,"available":True,"author":{"name":"Aldous Huxley","nationality":"British","birth_year":1894},"publisher":{"name":"Chatto & Windus","country":"UK"},"tags":["dystopia","classic","science-fiction","society"]},
    {"book_id":"bk-028","title":"The Alchemist","isbn":"978-0-06-112241-5","genre":"Fiction","language":"English","pages":208,"published_year":1988,"rating":4.3,"available":False,"author":{"name":"Paulo Coelho","nationality":"Brazilian","birth_year":1947},"publisher":{"name":"HarperOne","country":"USA"},"tags":["inspiration","journey","philosophy","fable"]},
    {"book_id":"bk-029","title":"One Hundred Years of Solitude","isbn":"978-0-06-088328-7","genre":"Fiction","language":"English","pages":417,"published_year":1967,"rating":4.6,"available":True,"author":{"name":"Gabriel Garcia Marquez","nationality":"Colombian","birth_year":1927},"publisher":{"name":"Harper & Row","country":"USA"},"tags":["magical-realism","classic","latin-america","family"]},
    {"book_id":"bk-030","title":"The Brothers Karamazov","isbn":"978-0-37-470278-3","genre":"Fiction","language":"English","pages":796,"published_year":1880,"rating":4.7,"available":True,"author":{"name":"Fyodor Dostoevsky","nationality":"Russian","birth_year":1821},"publisher":{"name":"Farrar, Straus and Giroux","country":"USA"},"tags":["classic","russian-literature","philosophy","family"]},
    {"book_id":"bk-031","title":"Anna Karenina","isbn":"978-0-14-303500-6","genre":"Fiction","language":"English","pages":864,"published_year":1878,"rating":4.5,"available":True,"author":{"name":"Leo Tolstoy","nationality":"Russian","birth_year":1828},"publisher":{"name":"Penguin Classics","country":"UK"},"tags":["classic","russian-literature","romance","tragedy"]},
    {"book_id":"bk-032","title":"Ulysses","isbn":"978-0-19-953434-4","genre":"Fiction","language":"English","pages":736,"published_year":1922,"rating":4.2,"available":False,"author":{"name":"James Joyce","nationality":"Irish","birth_year":1882},"publisher":{"name":"Shakespeare and Company","country":"France"},"tags":["modernism","classic","stream-of-consciousness","irish-literature"]},
    {"book_id":"bk-033","title":"The Grapes of Wrath","isbn":"978-0-14-303943-1","genre":"Fiction","language":"English","pages":479,"published_year":1939,"rating":4.5,"available":True,"author":{"name":"John Steinbeck","nationality":"American","birth_year":1902},"publisher":{"name":"Viking Press","country":"USA"},"tags":["classic","american-literature","depression-era","social-justice"]},
    {"book_id":"bk-034","title":"Moby Dick","isbn":"978-0-14-243724-4","genre":"Fiction","language":"English","pages":654,"published_year":1851,"rating":4.1,"available":True,"author":{"name":"Herman Melville","nationality":"American","birth_year":1819},"publisher":{"name":"Harper & Brothers","country":"USA"},"tags":["classic","adventure","sea","obsession"]},
    {"book_id":"bk-035","title":"Middlemarch","isbn":"978-0-14-143388-2","genre":"Fiction","language":"English","pages":904,"published_year":1871,"rating":4.5,"available":True,"author":{"name":"George Eliot","nationality":"British","birth_year":1819},"publisher":{"name":"Penguin Classics","country":"UK"},"tags":["classic","victorian","social-realism","british-literature"]},
    {"book_id":"bk-036","title":"The Stranger","isbn":"978-0-67-972020-1","genre":"Fiction","language":"English","pages":159,"published_year":1942,"rating":4.3,"available":True,"author":{"name":"Albert Camus","nationality":"French","birth_year":1913},"publisher":{"name":"Gallimard","country":"France"},"tags":["existentialism","absurdism","classic","philosophy"]},
    {"book_id":"bk-037","title":"Lolita","isbn":"978-0-67-972020-8","genre":"Fiction","language":"English","pages":336,"published_year":1955,"rating":4.2,"available":False,"author":{"name":"Vladimir Nabokov","nationality":"Russian","birth_year":1899},"publisher":{"name":"Olympia Press","country":"France"},"tags":["classic","controversial","literary-fiction","language"]},
    {"book_id":"bk-038","title":"The Sound and the Fury","isbn":"978-0-67-973224-2","genre":"Fiction","language":"English","pages":326,"published_year":1929,"rating":4.2,"available":True,"author":{"name":"William Faulkner","nationality":"American","birth_year":1897},"publisher":{"name":"Cape & Smith","country":"USA"},"tags":["classic","modernism","stream-of-consciousness","southern-gothic"]},
    {"book_id":"bk-039","title":"Beloved","isbn":"978-1-40-030215-5","genre":"Fiction","language":"English","pages":324,"published_year":1987,"rating":4.4,"available":True,"author":{"name":"Toni Morrison","nationality":"American","birth_year":1931},"publisher":{"name":"Knopf","country":"USA"},"tags":["slavery","american-history","magical-realism","literary-fiction"]},
    {"book_id":"bk-040","title":"The Road","isbn":"978-0-30-726543-2","genre":"Fiction","language":"English","pages":287,"published_year":2006,"rating":4.3,"available":True,"author":{"name":"Cormac McCarthy","nationality":"American","birth_year":1933},"publisher":{"name":"Knopf","country":"USA"},"tags":["post-apocalyptic","survival","father-son","literary-fiction"]},

    # ── Science Fiction & Fantasy (bk-041 to bk-060) ─────────────────────────
    {"book_id":"bk-041","title":"Dune","isbn":"978-0-44-101730-5","genre":"Science Fiction","language":"English","pages":896,"published_year":1965,"rating":4.9,"available":False,"author":{"name":"Frank Herbert","nationality":"American","birth_year":1920},"publisher":{"name":"Chilton Books","country":"USA"},"tags":["sci-fi","classic","epic","desert"]},
    {"book_id":"bk-042","title":"The Hitchhiker's Guide to the Galaxy","isbn":"978-0-34-539180-3","genre":"Science Fiction","language":"English","pages":224,"published_year":1979,"rating":4.8,"available":True,"author":{"name":"Douglas Adams","nationality":"British","birth_year":1952},"publisher":{"name":"Pan Books","country":"UK"},"tags":["sci-fi","comedy","classic","adventure"]},
    {"book_id":"bk-043","title":"Ender's Game","isbn":"978-0-76-530556-3","genre":"Science Fiction","language":"English","pages":352,"published_year":1985,"rating":4.7,"available":True,"author":{"name":"Orson Scott Card","nationality":"American","birth_year":1951},"publisher":{"name":"Tor Books","country":"USA"},"tags":["sci-fi","military","strategy","youth"]},
    {"book_id":"bk-044","title":"Foundation","isbn":"978-0-55-338020-5","genre":"Science Fiction","language":"English","pages":244,"published_year":1951,"rating":4.7,"available":True,"author":{"name":"Isaac Asimov","nationality":"American","birth_year":1920},"publisher":{"name":"Gnome Press","country":"USA"},"tags":["sci-fi","classic","empire","future"]},
    {"book_id":"bk-045","title":"Neuromancer","isbn":"978-0-44-151800-9","genre":"Science Fiction","language":"English","pages":271,"published_year":1984,"rating":4.4,"available":True,"author":{"name":"William Gibson","nationality":"American","birth_year":1948},"publisher":{"name":"Ace Books","country":"USA"},"tags":["cyberpunk","sci-fi","classic","hacking"]},
    {"book_id":"bk-046","title":"The Lord of the Rings","isbn":"978-0-61-864015-7","genre":"Fantasy","language":"English","pages":1216,"published_year":1954,"rating":4.9,"available":True,"author":{"name":"J.R.R. Tolkien","nationality":"British","birth_year":1892},"publisher":{"name":"Allen & Unwin","country":"UK"},"tags":["fantasy","classic","epic","adventure","mythology"]},
    {"book_id":"bk-047","title":"The Name of the Wind","isbn":"978-0-75-640474-1","genre":"Fantasy","language":"English","pages":662,"published_year":2007,"rating":4.7,"available":True,"author":{"name":"Patrick Rothfuss","nationality":"American","birth_year":1973},"publisher":{"name":"DAW Books","country":"USA"},"tags":["fantasy","magic","adventure","epic"]},
    {"book_id":"bk-048","title":"A Game of Thrones","isbn":"978-0-55-357340-3","genre":"Fantasy","language":"English","pages":694,"published_year":1996,"rating":4.7,"available":True,"author":{"name":"George R.R. Martin","nationality":"American","birth_year":1948},"publisher":{"name":"Bantam Books","country":"USA"},"tags":["fantasy","politics","war","epic"]},
    {"book_id":"bk-049","title":"The Way of Kings","isbn":"978-0-76-533655-2","genre":"Fantasy","language":"English","pages":1007,"published_year":2010,"rating":4.8,"available":False,"author":{"name":"Brandon Sanderson","nationality":"American","birth_year":1975},"publisher":{"name":"Tor Books","country":"USA"},"tags":["fantasy","epic","magic","world-building"]},
    {"book_id":"bk-050","title":"Harry Potter and the Philosopher's Stone","isbn":"978-0-74-750420-7","genre":"Fantasy","language":"English","pages":332,"published_year":1997,"rating":4.8,"available":True,"author":{"name":"J.K. Rowling","nationality":"British","birth_year":1965},"publisher":{"name":"Bloomsbury","country":"UK"},"tags":["fantasy","magic","youth","school"]},
    {"book_id":"bk-051","title":"The Hobbit","isbn":"978-0-54-792822-7","genre":"Fantasy","language":"English","pages":310,"published_year":1937,"rating":4.7,"available":True,"author":{"name":"J.R.R. Tolkien","nationality":"British","birth_year":1892},"publisher":{"name":"Allen & Unwin","country":"UK"},"tags":["fantasy","classic","adventure","quest"]},
    {"book_id":"bk-052","title":"American Gods","isbn":"978-0-38-053363-5","genre":"Fantasy","language":"English","pages":635,"published_year":2001,"rating":4.5,"available":True,"author":{"name":"Neil Gaiman","nationality":"British","birth_year":1960},"publisher":{"name":"William Morrow","country":"USA"},"tags":["fantasy","mythology","gods","america"]},
    {"book_id":"bk-053","title":"The Blade Itself","isbn":"978-0-57-508131-0","genre":"Fantasy","language":"English","pages":515,"published_year":2006,"rating":4.5,"available":False,"author":{"name":"Joe Abercrombie","nationality":"British","birth_year":1974},"publisher":{"name":"Gollancz","country":"UK"},"tags":["fantasy","grimdark","war","anti-hero"]},
    {"book_id":"bk-054","title":"Snow Crash","isbn":"978-0-55-338095-3","genre":"Science Fiction","language":"English","pages":440,"published_year":1992,"rating":4.5,"available":True,"author":{"name":"Neal Stephenson","nationality":"American","birth_year":1959},"publisher":{"name":"Bantam Books","country":"USA"},"tags":["sci-fi","cyberpunk","metaverse","hacking"]},
    {"book_id":"bk-055","title":"The Martian","isbn":"978-0-80-413902-1","genre":"Science Fiction","language":"English","pages":369,"published_year":2011,"rating":4.7,"available":True,"author":{"name":"Andy Weir","nationality":"American","birth_year":1972},"publisher":{"name":"Crown Publishers","country":"USA"},"tags":["sci-fi","survival","space","humor"]},
    {"book_id":"bk-056","title":"Recursion","isbn":"978-1-52-400591-8","genre":"Science Fiction","language":"English","pages":329,"published_year":2019,"rating":4.6,"available":True,"author":{"name":"Blake Crouch","nationality":"American","birth_year":1978},"publisher":{"name":"Crown Publishers","country":"USA"},"tags":["sci-fi","thriller","time-travel","memory"]},
    {"book_id":"bk-057","title":"Project Hail Mary","isbn":"978-0-59-313520-4","genre":"Science Fiction","language":"English","pages":476,"published_year":2021,"rating":4.9,"available":False,"author":{"name":"Andy Weir","nationality":"American","birth_year":1972},"publisher":{"name":"Ballantine Books","country":"USA"},"tags":["sci-fi","space","survival","alien"]},
    {"book_id":"bk-058","title":"The Three-Body Problem","isbn":"978-0-76-538955-8","genre":"Science Fiction","language":"English","pages":400,"published_year":2008,"rating":4.6,"available":True,"author":{"name":"Liu Cixin","nationality":"Chinese","birth_year":1963},"publisher":{"name":"Chongqing Publishing Group","country":"China"},"tags":["sci-fi","physics","alien-contact","chinese-literature"]},
    {"book_id":"bk-059","title":"Children of Time","isbn":"978-0-33-052656-8","genre":"Science Fiction","language":"English","pages":600,"published_year":2015,"rating":4.7,"available":True,"author":{"name":"Adrian Tchaikovsky","nationality":"British","birth_year":1972},"publisher":{"name":"Tor Books","country":"UK"},"tags":["sci-fi","evolution","spider","future"]},
    {"book_id":"bk-060","title":"The Left Hand of Darkness","isbn":"978-0-44-101285-1","genre":"Science Fiction","language":"English","pages":286,"published_year":1969,"rating":4.5,"available":True,"author":{"name":"Ursula K. Le Guin","nationality":"American","birth_year":1929},"publisher":{"name":"Ace Books","country":"USA"},"tags":["sci-fi","gender","classic","anthropology"]},

    # ── Non-Fiction: History, Science, Psychology (bk-061 to bk-080) ─────────
    {"book_id":"bk-061","title":"Sapiens","isbn":"978-0-06-231610-4","genre":"History","language":"English","pages":443,"published_year":2011,"rating":4.6,"available":True,"author":{"name":"Yuval Noah Harari","nationality":"Israeli","birth_year":1976},"publisher":{"name":"Harper Collins","country":"USA"},"tags":["history","anthropology","evolution","culture"]},
    {"book_id":"bk-062","title":"Homo Deus","isbn":"978-0-06-246431-6","genre":"History","language":"English","pages":450,"published_year":2015,"rating":4.4,"available":True,"author":{"name":"Yuval Noah Harari","nationality":"Israeli","birth_year":1976},"publisher":{"name":"Harper Collins","country":"USA"},"tags":["future","history","technology","humanity"]},
    {"book_id":"bk-063","title":"Guns, Germs, and Steel","isbn":"978-0-39-331755-8","genre":"History","language":"English","pages":480,"published_year":1997,"rating":4.5,"available":True,"author":{"name":"Jared Diamond","nationality":"American","birth_year":1937},"publisher":{"name":"W.W. Norton","country":"USA"},"tags":["history","anthropology","civilization","geography"]},
    {"book_id":"bk-064","title":"The Silk Roads","isbn":"978-1-40-880996-4","genre":"History","language":"English","pages":672,"published_year":2015,"rating":4.5,"available":False,"author":{"name":"Peter Frankopan","nationality":"British","birth_year":1971},"publisher":{"name":"Bloomsbury","country":"UK"},"tags":["history","trade","asia","civilization"]},
    {"book_id":"bk-065","title":"A Brief History of Time","isbn":"978-0-55-338016-3","genre":"Science","language":"English","pages":212,"published_year":1988,"rating":4.6,"available":False,"author":{"name":"Stephen Hawking","nationality":"British","birth_year":1942},"publisher":{"name":"Bantam Books","country":"USA"},"tags":["physics","cosmology","science","black-holes"]},
    {"book_id":"bk-066","title":"The Selfish Gene","isbn":"978-0-19-857519-1","genre":"Science","language":"English","pages":360,"published_year":1976,"rating":4.5,"available":True,"author":{"name":"Richard Dawkins","nationality":"British","birth_year":1941},"publisher":{"name":"Oxford University Press","country":"UK"},"tags":["biology","evolution","genetics","science"]},
    {"book_id":"bk-067","title":"Cosmos","isbn":"978-0-34-539135-3","genre":"Science","language":"English","pages":365,"published_year":1980,"rating":4.7,"available":True,"author":{"name":"Carl Sagan","nationality":"American","birth_year":1934},"publisher":{"name":"Random House","country":"USA"},"tags":["astronomy","science","cosmos","philosophy"]},
    {"book_id":"bk-068","title":"Thinking, Fast and Slow","isbn":"978-0-37-453355-7","genre":"Psychology","language":"English","pages":499,"published_year":2011,"rating":4.5,"available":False,"author":{"name":"Daniel Kahneman","nationality":"Israeli","birth_year":1934},"publisher":{"name":"Farrar, Straus and Giroux","country":"USA"},"tags":["psychology","behavioral-economics","decision-making"]},
    {"book_id":"bk-069","title":"The Body Keeps the Score","isbn":"978-0-14-312774-1","genre":"Psychology","language":"English","pages":464,"published_year":2014,"rating":4.8,"available":True,"author":{"name":"Bessel van der Kolk","nationality":"Dutch","birth_year":1943},"publisher":{"name":"Viking","country":"USA"},"tags":["trauma","psychology","healing","neuroscience"]},
    {"book_id":"bk-070","title":"Man's Search for Meaning","isbn":"978-0-80-701429-5","genre":"Psychology","language":"English","pages":165,"published_year":1946,"rating":4.7,"available":True,"author":{"name":"Viktor E. Frankl","nationality":"Austrian","birth_year":1905},"publisher":{"name":"Beacon Press","country":"USA"},"tags":["psychology","philosophy","holocaust","meaning"]},
    {"book_id":"bk-071","title":"The Power of Now","isbn":"978-1-57-731480-6","genre":"Self-Help","language":"English","pages":236,"published_year":1997,"rating":4.5,"available":True,"author":{"name":"Eckhart Tolle","nationality":"German","birth_year":1948},"publisher":{"name":"New World Library","country":"USA"},"tags":["mindfulness","spirituality","presence","self-help"]},
    {"book_id":"bk-072","title":"Atomic Habits","isbn":"978-0-73-521129-2","genre":"Self-Help","language":"English","pages":320,"published_year":2018,"rating":4.8,"available":True,"author":{"name":"James Clear","nationality":"American","birth_year":1986},"publisher":{"name":"Avery","country":"USA"},"tags":["habits","productivity","self-improvement","psychology"]},
    {"book_id":"bk-073","title":"Deep Work","isbn":"978-1-45-558669-1","genre":"Self-Help","language":"English","pages":296,"published_year":2016,"rating":4.6,"available":True,"author":{"name":"Cal Newport","nationality":"American","birth_year":1982},"publisher":{"name":"Grand Central Publishing","country":"USA"},"tags":["productivity","focus","work","digital-minimalism"]},
    {"book_id":"bk-074","title":"The 7 Habits of Highly Effective People","isbn":"978-0-74-320454-5","genre":"Self-Help","language":"English","pages":381,"published_year":1989,"rating":4.4,"available":True,"author":{"name":"Stephen R. Covey","nationality":"American","birth_year":1932},"publisher":{"name":"Free Press","country":"USA"},"tags":["productivity","leadership","habits","self-improvement"]},
    {"book_id":"bk-075","title":"Meditations","isbn":"978-0-14-044140-6","genre":"Philosophy","language":"English","pages":256,"published_year":180,"rating":4.8,"available":True,"author":{"name":"Marcus Aurelius","nationality":"Roman","birth_year":121},"publisher":{"name":"Penguin Classics","country":"UK"},"tags":["stoicism","philosophy","classic","mindfulness"]},
    {"book_id":"bk-076","title":"The Republic","isbn":"978-0-14-044914-3","genre":"Philosophy","language":"English","pages":416,"published_year":-380,"rating":4.4,"available":True,"author":{"name":"Plato","nationality":"Greek","birth_year":-428},"publisher":{"name":"Penguin Classics","country":"UK"},"tags":["philosophy","politics","justice","classic"]},
    {"book_id":"bk-077","title":"Thus Spoke Zarathustra","isbn":"978-0-14-044118-5","genre":"Philosophy","language":"English","pages":352,"published_year":1883,"rating":4.3,"available":False,"author":{"name":"Friedrich Nietzsche","nationality":"German","birth_year":1844},"publisher":{"name":"Penguin Classics","country":"UK"},"tags":["philosophy","existentialism","classic","german-literature"]},
    {"book_id":"bk-078","title":"The Subtle Art of Not Giving a F*ck","isbn":"978-0-06-245771-4","genre":"Self-Help","language":"English","pages":224,"published_year":2016,"rating":4.3,"available":True,"author":{"name":"Mark Manson","nationality":"American","birth_year":1984},"publisher":{"name":"HarperOne","country":"USA"},"tags":["self-help","mindset","values","humor"]},
    {"book_id":"bk-079","title":"The Coddling of the American Mind","isbn":"978-0-73-521130-8","genre":"Psychology","language":"English","pages":352,"published_year":2018,"rating":4.3,"available":True,"author":{"name":"Jonathan Haidt","nationality":"American","birth_year":1963},"publisher":{"name":"Penguin Press","country":"USA"},"tags":["psychology","education","society","resilience"]},
    {"book_id":"bk-080","title":"Outliers","isbn":"978-0-31-601792-3","genre":"Psychology","language":"English","pages":309,"published_year":2008,"rating":4.4,"available":True,"author":{"name":"Malcolm Gladwell","nationality":"Canadian","birth_year":1963},"publisher":{"name":"Little, Brown","country":"USA"},"tags":["success","psychology","culture","sociology"]},

    # ── Business & Economics (bk-081 to bk-095) ───────────────────────────────
    {"book_id":"bk-081","title":"The Lean Startup","isbn":"978-0-30-788791-7","genre":"Business","language":"English","pages":336,"published_year":2011,"rating":4.4,"available":True,"author":{"name":"Eric Ries","nationality":"American","birth_year":1978},"publisher":{"name":"Crown Business","country":"USA"},"tags":["startup","entrepreneurship","business","agile"]},
    {"book_id":"bk-082","title":"Zero to One","isbn":"978-0-80-413929-8","genre":"Business","language":"English","pages":224,"published_year":2014,"rating":4.5,"available":True,"author":{"name":"Peter Thiel","nationality":"American","birth_year":1967},"publisher":{"name":"Crown Business","country":"USA"},"tags":["startup","innovation","technology","entrepreneurship"]},
    {"book_id":"bk-083","title":"Good to Great","isbn":"978-0-06-662099-2","genre":"Business","language":"English","pages":320,"published_year":2001,"rating":4.3,"available":True,"author":{"name":"Jim Collins","nationality":"American","birth_year":1958},"publisher":{"name":"HarperBusiness","country":"USA"},"tags":["business","leadership","management","strategy"]},
    {"book_id":"bk-084","title":"The Hard Thing About Hard Things","isbn":"978-0-06-227320-6","genre":"Business","language":"English","pages":304,"published_year":2014,"rating":4.5,"available":True,"author":{"name":"Ben Horowitz","nationality":"American","birth_year":1966},"publisher":{"name":"HarperBusiness","country":"USA"},"tags":["startup","leadership","management","entrepreneurship"]},
    {"book_id":"bk-085","title":"Thinking in Systems","isbn":"978-1-60-358153-7","genre":"Science","language":"English","pages":240,"published_year":2008,"rating":4.6,"available":False,"author":{"name":"Donella H. Meadows","nationality":"American","birth_year":1941},"publisher":{"name":"Chelsea Green Publishing","country":"USA"},"tags":["systems-thinking","science","complexity","environment"]},
    {"book_id":"bk-086","title":"Never Split the Difference","isbn":"978-0-06-240780-1","genre":"Business","language":"English","pages":288,"published_year":2016,"rating":4.7,"available":True,"author":{"name":"Chris Voss","nationality":"American","birth_year":1957},"publisher":{"name":"HarperBusiness","country":"USA"},"tags":["negotiation","business","psychology","fbi"]},
    {"book_id":"bk-087","title":"The E-Myth Revisited","isbn":"978-0-88-730728-3","genre":"Business","language":"English","pages":288,"published_year":1995,"rating":4.3,"available":True,"author":{"name":"Michael E. Gerber","nationality":"American","birth_year":1936},"publisher":{"name":"HarperCollins","country":"USA"},"tags":["business","entrepreneurship","systems","small-business"]},
    {"book_id":"bk-088","title":"Principles","isbn":"978-1-50-116616-5","genre":"Business","language":"English","pages":592,"published_year":2017,"rating":4.5,"available":True,"author":{"name":"Ray Dalio","nationality":"American","birth_year":1949},"publisher":{"name":"Simon & Schuster","country":"USA"},"tags":["business","leadership","life","investing"]},
    {"book_id":"bk-089","title":"Freakonomics","isbn":"978-0-06-073132-0","genre":"Business","language":"English","pages":320,"published_year":2005,"rating":4.3,"available":True,"author":{"name":"Steven D. Levitt","nationality":"American","birth_year":1969},"publisher":{"name":"William Morrow","country":"USA"},"tags":["economics","data","society","counterintuitive"]},
    {"book_id":"bk-090","title":"The Innovator's Dilemma","isbn":"978-0-06-252091-2","genre":"Business","language":"English","pages":288,"published_year":1997,"rating":4.4,"available":False,"author":{"name":"Clayton M. Christensen","nationality":"American","birth_year":1952},"publisher":{"name":"Harvard Business Review Press","country":"USA"},"tags":["innovation","business","disruption","technology"]},
    {"book_id":"bk-091","title":"Measure What Matters","isbn":"978-0-52-551848-9","genre":"Business","language":"English","pages":320,"published_year":2018,"rating":4.4,"available":True,"author":{"name":"John Doerr","nationality":"American","birth_year":1951},"publisher":{"name":"Portfolio","country":"USA"},"tags":["okr","business","goal-setting","management"]},
    {"book_id":"bk-092","title":"High Output Management","isbn":"978-0-67-963225-1","genre":"Business","language":"English","pages":272,"published_year":1983,"rating":4.5,"available":True,"author":{"name":"Andrew S. Grove","nationality":"American","birth_year":1936},"publisher":{"name":"Random House","country":"USA"},"tags":["management","leadership","productivity","intel"]},
    {"book_id":"bk-093","title":"The Mom Test","isbn":"978-1-49-204214-4","genre":"Business","language":"English","pages":130,"published_year":2013,"rating":4.6,"available":True,"author":{"name":"Rob Fitzpatrick","nationality":"American","birth_year":1984},"publisher":{"name":"CreateSpace","country":"USA"},"tags":["startup","customer-development","entrepreneurship","validation"]},
    {"book_id":"bk-094","title":"Sprint","isbn":"978-1-50-113446-6","genre":"Business","language":"English","pages":288,"published_year":2016,"rating":4.3,"available":True,"author":{"name":"Jake Knapp","nationality":"American","birth_year":1978},"publisher":{"name":"Simon & Schuster","country":"USA"},"tags":["design-sprint","product","startup","google-ventures"]},
    {"book_id":"bk-095","title":"Inspired","isbn":"978-1-11-938713-4","genre":"Business","language":"English","pages":368,"published_year":2017,"rating":4.5,"available":False,"author":{"name":"Marty Cagan","nationality":"American","birth_year":1960},"publisher":{"name":"Wiley","country":"USA"},"tags":["product-management","startup","technology","leadership"]},

    # ── Biography & Memoir (bk-096 to bk-110) ────────────────────────────────
    {"book_id":"bk-096","title":"Steve Jobs","isbn":"978-1-45-160869-0","genre":"Biography","language":"English","pages":656,"published_year":2011,"rating":4.5,"available":True,"author":{"name":"Walter Isaacson","nationality":"American","birth_year":1952},"publisher":{"name":"Simon & Schuster","country":"USA"},"tags":["biography","apple","technology","entrepreneurship"]},
    {"book_id":"bk-097","title":"Elon Musk","isbn":"978-1-98-210728-3","genre":"Biography","language":"English","pages":688,"published_year":2023,"rating":4.4,"available":False,"author":{"name":"Walter Isaacson","nationality":"American","birth_year":1952},"publisher":{"name":"Simon & Schuster","country":"USA"},"tags":["biography","tesla","spacex","entrepreneurship"]},
    {"book_id":"bk-098","title":"The Diary of a Young Girl","isbn":"978-0-55-329698-1","genre":"Biography","language":"English","pages":283,"published_year":1947,"rating":4.8,"available":True,"author":{"name":"Anne Frank","nationality":"Dutch","birth_year":1929},"publisher":{"name":"Contact Publishing","country":"Netherlands"},"tags":["biography","holocaust","wwii","diary"]},
    {"book_id":"bk-099","title":"Long Walk to Freedom","isbn":"978-0-31-620084-5","genre":"Biography","language":"English","pages":656,"published_year":1994,"rating":4.8,"available":True,"author":{"name":"Nelson Mandela","nationality":"South African","birth_year":1918},"publisher":{"name":"Little, Brown","country":"USA"},"tags":["biography","apartheid","freedom","south-africa"]},
    {"book_id":"bk-100","title":"The Story of My Experiments with Truth","isbn":"978-0-80-700533-0","genre":"Biography","language":"English","pages":528,"published_year":1927,"rating":4.5,"available":True,"author":{"name":"Mahatma Gandhi","nationality":"Indian","birth_year":1869},"publisher":{"name":"Navajivan Publishing","country":"India"},"tags":["biography","india","nonviolence","philosophy"]},
    {"book_id":"bk-101","title":"I Know Why the Caged Bird Sings","isbn":"978-0-34-531997-9","genre":"Biography","language":"English","pages":304,"published_year":1969,"rating":4.6,"available":True,"author":{"name":"Maya Angelou","nationality":"American","birth_year":1928},"publisher":{"name":"Random House","country":"USA"},"tags":["biography","memoir","race","coming-of-age"]},
    {"book_id":"bk-102","title":"The Glass Castle","isbn":"978-0-74-321913-7","genre":"Biography","language":"English","pages":288,"published_year":2005,"rating":4.5,"available":False,"author":{"name":"Jeannette Walls","nationality":"American","birth_year":1960},"publisher":{"name":"Scribner","country":"USA"},"tags":["memoir","family","poverty","resilience"]},
    {"book_id":"bk-103","title":"Educated","isbn":"978-0-39-959050-4","genre":"Biography","language":"English","pages":334,"published_year":2018,"rating":4.7,"available":True,"author":{"name":"Tara Westover","nationality":"American","birth_year":1986},"publisher":{"name":"Random House","country":"USA"},"tags":["memoir","education","family","self-made"]},
    {"book_id":"bk-104","title":"Born a Crime","isbn":"978-0-39-958838-5","genre":"Biography","language":"English","pages":304,"published_year":2016,"rating":4.8,"available":True,"author":{"name":"Trevor Noah","nationality":"South African","birth_year":1984},"publisher":{"name":"Spiegel & Grau","country":"USA"},"tags":["memoir","apartheid","comedy","south-africa"]},
    {"book_id":"bk-105","title":"When Breath Becomes Air","isbn":"978-0-81-299367-5","genre":"Biography","language":"English","pages":228,"published_year":2016,"rating":4.7,"available":True,"author":{"name":"Paul Kalanithi","nationality":"American","birth_year":1977},"publisher":{"name":"Random House","country":"USA"},"tags":["memoir","medicine","death","meaning"]},
    {"book_id":"bk-106","title":"Becoming","isbn":"978-1-52-476313-8","genre":"Biography","language":"English","pages":448,"published_year":2018,"rating":4.7,"available":True,"author":{"name":"Michelle Obama","nationality":"American","birth_year":1964},"publisher":{"name":"Crown Publishing","country":"USA"},"tags":["memoir","politics","white-house","empowerment"]},
    {"book_id":"bk-107","title":"Bad Blood","isbn":"978-0-52-557342-3","genre":"Biography","language":"English","pages":352,"published_year":2018,"rating":4.7,"available":False,"author":{"name":"John Carreyrou","nationality":"American","birth_year":1971},"publisher":{"name":"Knopf","country":"USA"},"tags":["fraud","theranos","silicon-valley","journalism"]},
    {"book_id":"bk-108","title":"The Wright Brothers","isbn":"978-1-47-674521-3","genre":"History","language":"English","pages":336,"published_year":2015,"rating":4.6,"available":True,"author":{"name":"David McCullough","nationality":"American","birth_year":1933},"publisher":{"name":"Simon & Schuster","country":"USA"},"tags":["history","aviation","biography","invention"]},
    {"book_id":"bk-109","title":"Into the Wild","isbn":"978-0-38-548680-4","genre":"Biography","language":"English","pages":224,"published_year":1996,"rating":4.4,"available":True,"author":{"name":"Jon Krakauer","nationality":"American","birth_year":1954},"publisher":{"name":"Villard Books","country":"USA"},"tags":["memoir","adventure","nature","survival"]},
    {"book_id":"bk-110","title":"The Autobiography of Benjamin Franklin","isbn":"978-0-14-039052-1","genre":"Biography","language":"English","pages":146,"published_year":1791,"rating":4.3,"available":True,"author":{"name":"Benjamin Franklin","nationality":"American","birth_year":1706},"publisher":{"name":"Penguin Classics","country":"UK"},"tags":["biography","history","founding-fathers","self-made"]},
]


# ─────────────────────────────────────────────────────────────────────────────
# Auth
# ─────────────────────────────────────────────────────────────────────────────

_USERS = {
    "admin": {"password": "admin123", "role": "admin",  "token": "admin-token-abc"},
    "guest": {"password": "guest123", "role": "reader", "token": "guest-token-xyz"},
}
_VALID_TOKENS = {
    u["token"]: {"username": name, "role": u["role"]}
    for name, u in _USERS.items()
}

_bearer = HTTPBearer(auto_error=True)


def get_current_user(
    creds: HTTPAuthorizationCredentials = Security(_bearer),
) -> Dict:
    user = _VALID_TOKENS.get(creds.credentials)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_pagination(page: int, page_size: int, total: int) -> PaginationMeta:
    total_pages = math.ceil(total / page_size) if total else 0
    return PaginationMeta(
        page=page, page_size=page_size, total_items=total,
        total_pages=total_pages,
        has_next=page < total_pages, has_prev=page > 1,
        next_page=page + 1 if page < total_pages else None,
        prev_page=page - 1 if page > 1 else None,
    )


def _paginate(items: list, page: int, page_size: int) -> list:
    start = (page - 1) * page_size
    return items[start: start + page_size]


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Books Library API",
    description=(
        "Books library with Bearer token auth and pagination.\n\n"
        "**Credentials:** admin / admin123  |  guest / guest123"
    ),
    version="1.0.0",
)


# ── POST /auth/token ──────────────────────────────────────────────────────────

@app.post("/auth/token", response_model=TokenResponse, tags=["Auth"],
          summary="Get a Bearer token")
async def login(body: TokenRequest) -> TokenResponse:
    user = _USERS.get(body.username)
    if not user or user["password"] != body.password:
        raise HTTPException(status_code=401, detail="Incorrect username or password",
                            headers={"WWW-Authenticate": "Bearer"})
    return TokenResponse(access_token=user["token"], role=user["role"])


# ── GET /api/v1/books ─────────────────────────────────────────────────────────

@app.get("/api/v1/books", response_model=PaginatedBooks, tags=["Books"],
         summary="List all books (paginated)")
async def list_books(
    page:      int           = Query(default=1,  ge=1,        description="Page number (1-based)"),
    page_size: int           = Query(default=10, ge=1, le=50, description="Items per page (max 50)"),
    genre:     Optional[str] = Query(default=None, description="Filter by genre e.g. Technology, Fiction, Fantasy"),
    available: Optional[bool]= Query(default=None, description="Filter by availability: true or false"),
    language:  Optional[str] = Query(default=None, description="Filter by language e.g. English"),
    current_user: Dict       = Depends(get_current_user),
) -> PaginatedBooks:
    books = list(_BOOKS)
    filters: Dict[str, Any] = {}

    if genre:
        books = [b for b in books if b["genre"].lower() == genre.lower()]
        filters["genre"] = genre
    if available is not None:
        books = [b for b in books if b["available"] == available]
        filters["available"] = available
    if language:
        books = [b for b in books if b["language"].lower() == language.lower()]
        filters["language"] = language

    total     = len(books)
    page_data = _paginate(books, page, page_size)

    return PaginatedBooks(
        data=[Book(**b) for b in page_data],
        pagination=_build_pagination(page, page_size, total),
        filters=filters,
    )


# ── GET /api/v1/books/search ──────────────────────────────────────────────────

@app.get("/api/v1/books/search", response_model=PaginatedBooks, tags=["Books"],
         summary="Search books by title, author, or tag")
async def search_books(
    q:         str  = Query(..., min_length=1, description="Search term"),
    page:      int  = Query(default=1,  ge=1),
    page_size: int  = Query(default=10, ge=1, le=50),
    current_user: Dict = Depends(get_current_user),
) -> PaginatedBooks:
    term  = q.lower()
    books = [
        b for b in _BOOKS
        if term in b["title"].lower()
        or term in b["author"]["name"].lower()
        or any(term in tag for tag in b["tags"])
        or term in b["genre"].lower()
    ]
    page_data = _paginate(books, page, len(books) if not page_size else page_size)
    return PaginatedBooks(
        data=[Book(**b) for b in _paginate(books, page, page_size)],
        pagination=_build_pagination(page, page_size, len(books)),
        filters={"q": q},
    )


# ── GET /api/v1/books/{book_id} ───────────────────────────────────────────────

@app.get("/api/v1/books/{book_id}", response_model=Book, tags=["Books"],
         summary="Get a single book by ID")
async def get_book(
    book_id: str,
    current_user: Dict = Depends(get_current_user),
) -> Book:
    for b in _BOOKS:
        if b["book_id"] == book_id:
            return Book(**b)
    raise HTTPException(status_code=404, detail=f"Book not found: {book_id}")


# ── GET /api/v1/genres ────────────────────────────────────────────────────────

@app.get("/api/v1/genres", tags=["Meta"], summary="List all genres")
async def list_genres(
    current_user: Dict = Depends(get_current_user),
) -> JSONResponse:
    genres = sorted({b["genre"] for b in _BOOKS})
    return JSONResponse({"genres": genres, "total": len(genres)})


# ── GET /health ───────────────────────────────────────────────────────────────

@app.get("/health", include_in_schema=False)
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok", "total_books": len(_BOOKS)})

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8090)